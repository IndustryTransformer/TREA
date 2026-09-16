"""Synthetic-prior pretraining spike: does TabPFN-style synthetic data transfer into
TREA's single-row encoder via plain pretrain -> few-shot finetune?

Context (2026-09-16): pretrained ICL models (TabICL) beat trees on 36/51 of our
classification tables (logs/icl_vs_trees.json) where from-scratch TREA managed 12/51.
Real-corpus pretraining saturated at K~2-4 schemas (schema_scaling_results.json). The
open question before funding a GAUGE-scale build: is the missing ingredient just
*synthetic-prior scale*, or the *in-context objective*? This spike isolates the first.

Pretraining data: TabICL's open prior generator (`tabicl.prior.PriorDataset`,
mix_scm: MLP/tree structural causal models), regression targets, <=MAX_FEAT features.
Column identity: anonymous slots "f0..f{MAX_FEAT-1}", randomly permuted per table.
Objective (--objective):
  sup : one shared head predicts the (standardized) synthetic target from features
        (TabPFN's target without in-context rows).
  mtm : mask 30% of cells (target included as a column), per-token reconstruction.

Evaluation = scripts/schema_scaling.py protocol: 5 held-out real tables, 200 labels,
best-val finetune, test RMSE in target units. Arms per held-out:
  scratch_index / scratch_semantic : fresh encoder, few-shot only
  xgb                              : the tree bar
  synth{N}_index / synth{N}_semantic: encoder pretrained on N synthetic tables, fresh
        column identities on the held-out (index = the charitable control from
        schema_scaling; semantic = frozen-MiniLM names, untrained projection)
Reference (printed): transfer_index / transfer_semantic @K=12 real tables.

== PRE-REGISTERED (locked before results) ==
Synthetic-prior transfer VALIDATED iff at the largest N, best(synth_index,
synth_semantic):
  (1) < the matching scratch arm on >=4/5 held-out,
  (2) < transfer_index@K=12 (real-corpus backbone) on >=4/5 held-out,
  (3) improves from smallest N to largest N on >=3/5.
Beating XGB is NOT required here (that is the ICL-objective + scale question).
KILL (for plain pretrain->finetune on synthetic priors) if (1) or (2) fails: the
benefit then needs the in-context objective, which is a different architecture bet.

Run:  PYTHONPATH=. uv run python scripts/synthetic_pretrain_spike.py > logs/synth_spike.log 2>&1
"""

import argparse
import json
import os
import time
import warnings

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
torch.set_float32_matmul_precision("medium")

from scripts.schema_scaling import (  # noqa: E402
    D_MODEL,
    DEV,
    SharedModel,
    finetune_heldout,
    load_corpus,
    standardize,
    xgb_rmse,
)
from trea.models.column_embeddings import (  # noqa: E402
    IndexColumnEmbedder,
    SemanticColumnEmbedder,
)
from trea.training.single_row import copy_encoder_weights  # noqa: E402


MAX_FEAT = 30
SLOTS = [f"f{i}" for i in range(MAX_FEAT + 1)]  # +1 slot for the target column (mtm)
OUT = Path(__file__).resolve().parent.parent / "logs" / "synth_spike.json"
REF = Path(__file__).resolve().parent.parent / "schema_scaling_results.json"
HELD = [
    "california_housing",
    "concrete_compressive_strength",
    "abalone",
    "airfoil_self_noise",
    "Bike_Sharing_Demand",
]


def prior_loader(tables_per_batch, workers, seed):
    from tabicl.prior._dataset import PriorDataset

    torch.manual_seed(seed)
    ds = PriorDataset(
        regression=True,
        batch_size=tables_per_batch,
        batch_size_per_gp=4,
        min_features=2,
        max_features=MAX_FEAT,
        min_seq_len=64,
        max_seq_len=512,
        log_seq_len=True,
        seq_len_per_gp=False,  # dense [B, seq, MAX_FEAT] per batch (one seq_len)
        prior_type="mix_scm",
        n_jobs=1,
        device="cpu",
    )
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=workers,
        prefetch_factor=4 if workers > 0 else None,
        persistent_workers=workers > 0,
        worker_init_fn=lambda w: torch.manual_seed(seed * 1000 + w),
    )


class SynthPretrainer(nn.Module):
    """SharedModel-compatible encoder + objective-specific head for synthetic tables."""

    def __init__(self, objective):
        super().__init__()
        self.objective = objective
        emb = IndexColumnEmbedder(SLOTS, D_MODEL)
        self.shared = SharedModel(emb, ["synth"])  # encoder + pooling + 1 head (sup)
        self.tok_head = nn.Linear(D_MODEL, 1)  # per-token reconstruction (mtm)

    def loss(self, X, y, rng):
        # X [B, S, MAX_FEAT] (padded cols are zeros), y [B, S]; d = active features.
        B, S, _ = X.shape
        losses = []
        for b in range(B):
            d = int((X[b].abs().sum(0) > 0).sum().item())
            d = max(d, 2)
            xb = X[b, :, :d]
            perm = rng.permutation(len(SLOTS))
            if self.objective == "sup":
                names = [SLOTS[i] for i in perm[:d]]
                pred = self.shared("synth", xb, names)
                losses.append(((pred.squeeze(-1) - y[b]) ** 2).mean())
            else:
                full = torch.cat([xb, y[b].unsqueeze(1)], dim=1)  # target as a column
                names = [SLOTS[i] for i in perm[: d + 1]]
                mask = torch.rand(full.shape, device=full.device) > 0.7
                inp = full.clone()
                inp[mask] = float("nan")  # native missing == masked
                enc = self.shared.encoder
                enc.col_texts = names
                enc.numeric_texts = names
                h = enc(inp, None)  # [S, d+1, D]
                rec = self.tok_head(h).squeeze(-1)
                losses.append(((rec - full) ** 2)[mask].mean())
        return torch.stack(losses).mean()


def pretrain_synthetic(objective, n_tables, tables_per_batch, workers, lr, seed):
    model = SynthPretrainer(objective).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n_steps = max(1, n_tables // tables_per_batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps, pct_start=0.05
    )
    rng = np.random.RandomState(seed)
    it = iter(prior_loader(tables_per_batch, workers, seed))
    t0, run = time.perf_counter(), 0.0
    model.train()
    for step in range(n_steps):
        X, y, _d, _s, _t = next(it)
        X, y = X.to(DEV).float().clamp(-8, 8), y.to(DEV).float().clamp(-8, 8)
        X = torch.nan_to_num(X, nan=0.0, posinf=8.0, neginf=-8.0)
        loss = model.loss(X, y, rng)
        if not torch.isfinite(loss):
            skipped = globals().setdefault("_SKIPPED", [0])
            skipped[0] += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            print(
                f"    [{objective} N={n_tables}] step {step + 1}/{n_steps} "
                f"loss {run:.4f} skipped {globals().get('_SKIPPED', [0])[0]} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    return model.shared.encoder


def fresh_model(etype, descr):
    emb = (
        SemanticColumnEmbedder(descr, D_MODEL)
        if etype == "semantic"
        else IndexColumnEmbedder(descr, D_MODEL)
    )
    return SharedModel(emb, ["held"])


def eval_arm(encoder, etype, held, splits, descrs, n_labels, ft_ep, ft_lr, seeds):
    out = []
    for seed in range(seeds):
        m = fresh_model(etype, descrs[held["name"]])
        if encoder is not None:
            copy_encoder_weights(encoder, m.encoder, include_col_embedder=False)
        # finetune_heldout keys the head by dataset name -> alias
        m.heads[held["name"]] = m.heads["held"]
        out.append(
            finetune_heldout(
                m, held, splits, descrs, n_labels, ft_ep, ft_lr, splits[held["name"]][2], seed
            )
        )
    return float(np.mean(out)), float(np.std(out))


def reference():
    if not REF.exists():
        return {}
    r = json.load(open(REF))
    kmax = max(x["K"] for x in r)
    ref = {}
    for x in r:
        if x["K"] == kmax and x["arm"] in ("transfer_index", "transfer_semantic"):
            ref.setdefault(x["held_out"], {})[x["arm"]] = x["rmse"]
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["sup", "mtm"], default="sup")
    ap.add_argument("--n-tables", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--tables-per-batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pre-lr", type=float, default=5e-4)
    ap.add_argument("--n-labels", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ft_ep, ft_lr = (3, 1e-3) if a.smoke else (80, 5e-4)
    if a.smoke:
        a.n_tables, a.seeds, a.workers = [64], 1, 0
        held_names = HELD[:1]
    else:
        held_names = HELD

    corpus = load_corpus()
    by_name = {d["name"]: d for d in corpus}
    descrs = {d["name"]: d["descr"] for d in corpus}
    splits = {}
    for d in corpus:
        if d["name"] not in held_names:
            continue
        tr, te = train_test_split(d["df"], test_size=0.3, random_state=42)
        (Xtr, ytr), (Xte, yte), tsd = standardize(tr, te, d["feat"])
        splits[d["name"]] = ((Xtr, ytr), (Xte, yte), tsd)
    ref = reference()
    print(
        f"device={DEV} objective={a.objective} N={a.n_tables} held={held_names} "
        f"n_labels={a.n_labels} seeds={a.seeds}",
        flush=True,
    )

    results = {"objective": a.objective, "held": {}}
    # K-independent baselines
    for hn in held_names:
        h = by_name[hn]
        row = {"ref": ref.get(hn, {})}
        row["xgb"] = float(
            np.mean([xgb_rmse(h, splits, a.n_labels, splits[hn][2], s) for s in range(a.seeds)])
        )
        for et in ("index", "semantic"):
            row[f"scratch_{et}"] = eval_arm(
                None, et, h, splits, descrs, a.n_labels, ft_ep, ft_lr, a.seeds
            )
        results["held"][hn] = row
        print(
            f"[{hn}] xgb {row['xgb']:.3f} | scratch_index {row['scratch_index'][0]:.3f} "
            f"scratch_semantic {row['scratch_semantic'][0]:.3f} | ref@K12 "
            f"idx {row['ref'].get('transfer_index', float('nan')):.3f} "
            f"sem {row['ref'].get('transfer_semantic', float('nan')):.3f}",
            flush=True,
        )
    OUT.write_text(json.dumps(results, indent=2))

    for N in a.n_tables:
        print(f"\n=== pretraining on {N} synthetic tables ({a.objective}) ===", flush=True)
        enc = pretrain_synthetic(
            a.objective, N, a.tables_per_batch, a.workers, a.pre_lr, seed=N
        )
        for hn in held_names:
            h = by_name[hn]
            for et in ("index", "semantic"):
                results["held"][hn][f"synth{N}_{et}"] = eval_arm(
                    enc, et, h, splits, descrs, a.n_labels, ft_ep, ft_lr, a.seeds
                )
            r = results["held"][hn]
            print(
                f"  [{hn}] synth{N}: index {r[f'synth{N}_index'][0]:.3f}±{r[f'synth{N}_index'][1]:.3f} "
                f"semantic {r[f'synth{N}_semantic'][0]:.3f}±{r[f'synth{N}_semantic'][1]:.3f} "
                f"| scratch idx/sem {r['scratch_index'][0]:.3f}/{r['scratch_semantic'][0]:.3f} "
                f"| ref idx {r['ref'].get('transfer_index', float('nan')):.3f} | xgb {r['xgb']:.3f}",
                flush=True,
            )
            OUT.write_text(json.dumps(results, indent=2))

    # ----- verdict -----
    Ns = a.n_tables
    Nmax, Nmin = Ns[-1], Ns[0]
    c1 = c2 = c3 = 0
    print("\n=== VERDICT (pre-registered) ===", flush=True)
    for hn, r in results["held"].items():
        best_max = min(r[f"synth{Nmax}_index"][0], r[f"synth{Nmax}_semantic"][0])
        best_et = "index" if r[f"synth{Nmax}_index"][0] <= r[f"synth{Nmax}_semantic"][0] else "semantic"
        scratch = r[f"scratch_{best_et}"][0]
        ref_idx = r["ref"].get("transfer_index", float("inf"))
        best_min = min(r[f"synth{Nmin}_index"][0], r[f"synth{Nmin}_semantic"][0])
        p1, p2, p3 = best_max < scratch, best_max < ref_idx, best_max < best_min
        c1, c2, c3 = c1 + p1, c2 + p2, c3 + p3
        print(
            f"  {hn:30s} synth@{Nmax} {best_max:.3f} ({best_et}) | <scratch {p1} | "
            f"<realK12idx {p2} ({ref_idx:.3f}) | improves {Nmin}->{Nmax} {p3} | xgb {r['xgb']:.3f}",
            flush=True,
        )
    n = len(results["held"])
    ok = c1 >= 4 and c2 >= 4 and c3 >= 3 if n == 5 else None
    print(
        f"  (1) beats scratch {c1}/{n} | (2) beats real-corpus K=12 backbone {c2}/{n} | "
        f"(3) scales with N {c3}/{n} -> {'VALIDATED' if ok else ('KILL' if ok is False else 'n/a (smoke)')}",
        flush=True,
    )
    results["verdict"] = {"beats_scratch": c1, "beats_realK12": c2, "scales": c3, "n": n}
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nresults -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
