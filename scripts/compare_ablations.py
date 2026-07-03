"""Compare ablation runs produced by scripts/run_ablations.sh.

Reads results/ablations/*.json (each the full tabular_vs_trees output for one
flag configuration) and prints, per task, the mean deep-model score for each
config and its delta vs `baseline` on the datasets they share. Trees (XGB/RF)
from the baseline are shown as the reference bar. Deltas are attribution: since
each ablation flips one flag from baseline, its delta isolates that lever.

    uv run python scripts/compare_ablations.py

Classification is macro-F1 (higher better); regression is RMSE (lower better),
so a "better" flag moves the score in the task-appropriate direction.
"""

import json

from pathlib import Path


ABL = Path(__file__).resolve().parent.parent / "results" / "ablations"


def _load(name):
    p = ABL / f"{name}.json"
    if not p.exists():
        return None
    return {r["name"]: r for r in json.loads(p.read_text())}


def _mean(records, names, key):
    return sum(records[n][key] for n in names) / len(names)


def main():
    base = _load("baseline")
    if base is None:
        raise SystemExit(
            f"no baseline at {ABL / 'baseline.json'} -- run run_ablations.sh first"
        )
    others = sorted(p.stem for p in ABL.glob("*.json") if p.stem != "baseline")

    for task, higher in (("classification", True), ("regression", False)):
        base_names = [n for n, r in base.items() if r["task"] == task]
        if not base_names:
            continue
        metric = "macro-F1" if higher else "RMSE"
        arrow = "higher=better" if higher else "lower=better"
        print(f"\n=== {task}  ({metric}, {arrow}; n={len(base_names)} datasets) ===")

        b_deep = _mean(base, base_names, "deep")
        b_xgb = _mean(base, base_names, "xgb")
        b_rf = _mean(base, base_names, "rf")
        n_collapsed = sum(bool(base[n].get("collapsed")) for n in base_names)
        print(
            f"  {'baseline (deep)':26s} {b_deep:.4f}"
            f"   ref: xgb {b_xgb:.4f} | rf {b_rf:.4f}"
            + (f" | collapsed:{n_collapsed}" if higher else "")
        )

        for name in others:
            other = _load(name)
            shared = [n for n in base_names if n in other and other[n]["task"] == task]
            if not shared:
                continue
            o_deep = _mean(other, shared, "deep")
            b_on_shared = _mean(base, shared, "deep")
            delta = o_deep - b_on_shared
            better = (delta > 0) == higher
            tag = (
                "better"
                if better and abs(delta) > 1e-4
                else ("worse" if abs(delta) > 1e-4 else "~same")
            )
            extra = ""
            if higher:
                oc = sum(bool(other[n].get("collapsed")) for n in shared)
                extra = f"  collapsed:{oc}"
            print(
                f"  {name:26s} {o_deep:.4f}   Δ {delta:+.4f}  {tag:6s}"
                f"  (n={len(shared)}){extra}"
            )


if __name__ == "__main__":
    main()
