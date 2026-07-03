#!/usr/bin/env bash
# One-flag-at-a-time ablations over the merged WO-1..7 levers.
#
# Attributes each change's effect by flipping a SINGLE flag from the honest
# baseline (WO-1/2/3 are always on: logit-adjusted CE, QK/pre-norm + warmup,
# AdamW param-groups + macro-F1 model selection). Each run is a FULL multi-seed
# benchmark over the 89-dataset corpus -> SLOW; run on a GPU box, not in-chat.
#
#   bash scripts/run_ablations.sh                    # all ablations, 3 seeds
#   SEEDS=1 bash scripts/run_ablations.sh            # quick pass
#   ONLY=baseline,ensemble_k32 bash scripts/run_ablations.sh
#
# Results land in results/ablations/<name>.json. Compare with:
#   uv run python scripts/compare_ablations.py
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/ablations
mkdir -p "$OUT"
RESULTS=tabular_vs_trees_results.json
SEEDS=${SEEDS:-3}

want () { [[ -z "${ONLY:-}" ]] || [[ ",${ONLY}," == *",$1,"* ]]; }

run () {  # name  extra-args...
  local name="$1"; shift
  want "$name" || return 0
  echo "=================== ablation: ${name}  [$*] ==================="
  uv run python scripts/tabular_vs_trees.py --seeds "$SEEDS" "$@"
  cp "$RESULTS" "$OUT/${name}.json"
  echo "saved -> $OUT/${name}.json"
}

# Honest baseline (all WO-1/2/3 defaults; MSE head, identity encoder, no ensemble).
run baseline
# WO-1: logit adjustment OFF (classification) -> expect regressions on imbalanced
#       sets and the six known collapse datasets (pc3, pc4, ilpd, ozone, blood, connect_4).
run no_logit_adjust        --tasks classification --logit-adjustment-tau 0
# WO-5: HL-Gauss binned regression head (regression only).
run hl_gauss               --tasks regression --head hl_gauss
# WO-6: dual numeric value encoder (both tasks).
run value_dual             --value-encoder dual
# WO-6: dual encoder + hadamard fusion (both tasks).
run value_dual_hadamard    --value-encoder dual --fusion hadamard
# WO-7: TabM-mini BatchEnsemble, k=32 (both tasks).
run ensemble_k32           --ensemble-k 32

echo
echo "All ablations done. Compare with: uv run python scripts/compare_ablations.py"
