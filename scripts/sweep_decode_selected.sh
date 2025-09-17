#!/usr/bin/env bash
# Selective sweep with dynamic GPU scheduling (SMC + Default)
#
# Runs custom SMC decoding across the scoring/group/aggregation/window/top-k sweeps defined below.
# The default-decoding block remains available but is disabled by default (DEFAULT_TEMPS is empty).
#
# Notes
# - GPU assignment mirrors scripts/sweep_decode_smc.sh (one job per GPU slot).
# - Remaining configuration values fall back to config/decode_eval.yaml.
# - For the default runs, run_name varies only by tmp (temperature).
#
# Usage
#   bash scripts/sweep_decode_selected.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/sweep_decode_selected.sh

set -euo pipefail

# Fixed model list (override by setting MODEL_NAMES env if desired)
MODEL_NAMES=(${MODEL_NAMES:-"Qwen/Qwen2.5-0.5B"})

# Custom SMC confidence sweeps (values applied on top of config/decode_eval.yaml)
SCORINGS=("entropy" "logprob" "prob")
declare -A TOPK_MAP=(
  [entropy]="20 40"
  [logprob]="20 40"
  [prob]="1"
)
GROUPS=("mean" "geo")
AGGREGATIONS=("mean" "prod" "min" "last")
WINDOW_SIZES=(1 8 32 64 128 256 512)

# Default-decoding temperatures
DEFAULT_TEMPS=() 

# ---------------- GPU scheduler (same behavior as sweep_decode_smc.sh) ----------------
detect_visible_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    if [ ${#GPUS[@]} -gt 0 ]; then return; fi
  fi
  local count
  count=$(python3 - << 'PY' 2>/dev/null || echo 0
import sys
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
  if [ "$count" -gt 0 ] 2>/dev/null; then
    GPUS=()
    for ((i=0; i<count; i++)); do GPUS+=("$i"); done
  else
    GPUS=(0)
  fi
}

detect_visible_gpus
NGPUS=${#GPUS[@]}
declare -a PIDS
echo "Detected GPUs: ${GPUS[*]} (NGPUS=${NGPUS})"

submit_job() {
  local gpu_index=$1; shift
  local -a job_cmd=("$@")
  CUDA_VISIBLE_DEVICES="${GPUS[$gpu_index]}" "${job_cmd[@]}" &
  local pid=$!
  PIDS[$gpu_index]=$pid
  echo "Launched on GPU ${GPUS[$gpu_index]} (slot ${gpu_index}), PID ${pid}"
}

find_free_gpu() {
  local idx
  for ((idx=0; idx<NGPUS; idx++)); do
    local pid=${PIDS[$idx]:-}
    if [ -z "$pid" ]; then
      echo $idx; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      PIDS[$idx]=
      echo $idx; return 0
    fi
  done
  echo -1
}

# ---------------- Run name helpers ----------------
sanitize_model_name() { echo -n "$1" | sed 's#/#-#g'; }

build_run_name_smc() {
  local model_name="$1"; local scoring="$2"; local topk="$3"; local group="$4"; local agg="$5"; local win="$6"
  local safe_model
  safe_model=$(sanitize_model_name "$model_name")
  echo "smc_${safe_model}_score-${scoring}_topk${topk}_group-${group}_agg-${agg}_win${win}"
}

# For default runs, vary name only by tmp (temperature)
build_run_name_default() { echo "default_tmp$1"; }

# ---------------- Submit jobs ----------------
jobs_submitted=0

# 1) SMC custom decoding
for model_name in "${MODEL_NAMES[@]}"; do
  for scoring in "${SCORINGS[@]}"; do
    read -r -a scoring_topks <<< "${TOPK_MAP[$scoring]}"
    for topk in "${scoring_topks[@]}"; do
      for group in "${GROUPS[@]}"; do
        for agg in "${AGGREGATIONS[@]}"; do
          for win in "${WINDOW_SIZES[@]}"; do
            name="$(build_run_name_smc "$model_name" "$scoring" "$topk" "$group" "$agg" "$win")"
            cmd=(uv run python3 evaluate-decode.py \
              eval.run_default=false eval.run_custom=true \
              model.model_name="${model_name}" \
              custom_decode.smc_topk=${topk} \
              custom_decode.smc_confidence_window_size=${win} \
              custom_decode.confidence.scoring=${scoring} \
              custom_decode.confidence.group=${group} \
              custom_decode.confidence.aggregation=${agg} \
              wandb.run_name=${name})

            while true; do
              slot=$(find_free_gpu)
              if [ "$slot" -ge 0 ]; then
                echo "Running (SMC): ${cmd[*]}"
                submit_job "$slot" "${cmd[@]}"
                jobs_submitted=$((jobs_submitted+1))
                break
              fi
              sleep 2
            done
          done
        done
      done
    done
  done
done

# 2) Default decoding with fast inference
for model_name in "${MODEL_NAMES[@]}"; do
  for tmp in "${DEFAULT_TEMPS[@]}"; do
    name="$(build_run_name_default "$tmp")"
    cmd=(uv run python3 evaluate-decode.py \
      eval.run_default=true eval.run_custom=false \
      model.fast_inference=true \
      model.model_name="${model_name}" \
      default_decode.temperature=${tmp} \
      wandb.run_name=${name})

    while true; do
      slot=$(find_free_gpu)
      if [ "$slot" -ge 0 ]; then
        echo "Running (DEFAULT): ${cmd[*]}"
        submit_job "$slot" "${cmd[@]}"
        jobs_submitted=$((jobs_submitted+1))
        break
      fi
      sleep 2
    done
  done
done

# Wait for all jobs to finish
for pid in "${PIDS[@]}"; do
  if [ -n "${pid:-}" ]; then
    wait "$pid"
  fi
done

echo "Completed ${jobs_submitted} jobs."
