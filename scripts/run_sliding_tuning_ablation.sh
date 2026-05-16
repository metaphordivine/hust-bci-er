#!/usr/bin/env bash
# Master runner: coarse grid -> fine grid -> ablation for sliding window models.
# Usage: bash scripts/run_sliding_tuning_ablation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${HUST_BCI_ER_DATA_ROOT:?Set HUST_BCI_ER_DATA_ROOT to the HUST BCI ER train data root}"
DATA_ROOT="$HUST_BCI_ER_DATA_ROOT"
SEARCH_SPACE="$ROOT/configs/search/spaces/sliding_baseline.yaml"
ABLATION_SPEC="$ROOT/configs/search/ablations/sliding_standard.yaml"
SEED=42
DEVICE=auto
SMOKE_EPOCHS=1

MODELS=(
  "sliding_window_eegnet"
  "ea_deformer"
  "sliding_window_deformer_lite"
  "sliding_window_conformer_lite"
  "sliding_window_srfnet"
  "sliding_window_cbramod"
  "sliding_window_fbstcnet"
  "sliding_window_shallow_conv_net"
)

# Ablation spec per model (different for sliding vs ea)
declare -A ABLATION_SPECS=(
  ["sliding_window_eegnet"]="$ROOT/configs/search/ablations/sliding_standard.yaml"
  ["sliding_window_deformer_lite"]="$ROOT/configs/search/ablations/sliding_standard.yaml"
  ["sliding_window_conformer_lite"]="$ROOT/configs/search/ablations/sliding_standard.yaml"
  ["sliding_window_srfnet"]="$ROOT/configs/search/ablations/sliding_standard.yaml"
  ["sliding_window_cbramod"]="$ROOT/configs/search/ablations/sliding_standard.yaml"
  ["sliding_window_fbstcnet"]="$ROOT/configs/search/ablations/sliding_standard.yaml"
  ["sliding_window_shallow_conv_net"]="$ROOT/configs/search/ablations/sliding_standard.yaml"
  ["ea_deformer"]="$ROOT/configs/search/ablations/ea_deformer.yaml"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_stage() {
  local model="$1" stage="$2"
  local route="$ROOT/configs/routes/models/${model}.yaml"
  local run_dir="$ROOT/outputs/${model}/hparam_search"
  log "=== $model :: $stage grid ==="
  python "$ROOT/scripts/hparam_search.py" \
    --route "$route" \
    --stage "$stage" \
    --search-space "$SEARCH_SPACE" \
    --run-dir "$run_dir" \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --smoke-epochs "$SMOKE_EPOCHS"
}

run_ablation() {
  local model="$1"
  local route="$ROOT/configs/routes/models/${model}.yaml"
  local spec="${ABLATION_SPECS[$model]}"
  local run_dir="$ROOT/outputs/${model}/ablation"
  log "=== $model :: ablation (spec: ${spec##*/}) ==="
  python "$ROOT/scripts/run_ablation.py" \
    --route "$route" \
    --ablations "$spec" \
    --run-dir "$run_dir" \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --smoke-epochs "$SMOKE_EPOCHS"
}

log "===== SLIDING WINDOW TUNING + ABLATION ====="
log "Data root: $DATA_ROOT"
log "Models: ${MODELS[*]}"
log ""

for model in "${MODELS[@]}"; do
  log ""
  log "############################################"
  log "  MODEL: $model"
  log "############################################"

  route="$ROOT/configs/routes/models/${model}.yaml"
  if [[ ! -f "$route" ]]; then
    log "Skipping $model: route config not found at $route"
    continue
  fi

  # 1. Coarse grid
  run_stage "$model" coarse

  # 2. Fine grid (uses coarse results)
  run_stage "$model" fine

  # 3. Ablation
  run_ablation "$model"

  log "$model complete."
done

log ""
log "===== ALL MODELS COMPLETE ====="
