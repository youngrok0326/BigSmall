#!/usr/bin/env bash
# Selective sweep with dynamic GPU scheduling (SMC + Default)
#
# Runs two sets:
# 1) Custom SMC decoding on the exact (win, ess, tmp, N) combos below.
# 2) Default decoding for temperatures {0.9, 0.8, 0.7, 0.6, 0.5}.
#
# Notes
# - GPU assignment mirrors scripts/sweep_decode_smc.sh (one job per GPU slot).
# - GEN_GROUPS is fixed to (16 16) as requested.
# - For the default runs, run_name varies only by tmp (temperature).
#
# Usage
#   bash scripts/sweep_decode_selected.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/sweep_decode_selected.sh

set -euo pipefail

# Fixed model list (override by setting MODEL_NAMES env if desired)
MODEL_NAMES=(${MODEL_NAMES:-"Qwen/Qwen2.5-0.5B"})

# Fixed GEN_GROUPS: (batch_size_groups num_generations)
G=64
N=16

# Exact SMC combos: (win ess tmp N) — defined as an array (safe under set -e)
SMC_COMBOS=(
"1024 0.5 0.9 16"
"512 0.5 0.9 16"
"256 0.5 0.9 16"
"128 0.5 0.9 16"
"64 0.5 0.9 16"
)

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
  local model_name="$1"; local tmp="$2"; local ess="$3"; local win="$4"
  local safe_model
  safe_model=$(sanitize_model_name "$model_name")
  echo "smc_${safe_model}_N${N}_tmp${tmp}_ess${ess}_win${win}"
}

# For default runs, vary name only by tmp (temperature)
build_run_name_default() { echo "default_tmp$1"; }

# ---------------- Submit jobs ----------------
jobs_submitted=0

# 1) SMC custom decoding
for model_name in "${MODEL_NAMES[@]}"; do
  for line in "${SMC_COMBOS[@]}"; do
    read -r win ess tmp n_from_combo <<< "$line"
    # Enforce fixed GEN_GROUPS; ignore n_from_combo if different
    name="$(build_run_name_smc "$model_name" "$tmp" "$ess" "$win")"
    cmd=(uv run python3 evaluate-decode.py \
      eval.run_default=false eval.run_custom=true \
      model.model_name="${model_name}" \
      eval.batch_size_groups=${G} \
      eval.num_generations=${N} \
      custom_decode.temperature=${tmp} \
      custom_decode.smc_resample_threshold=${ess} \
      custom_decode.smc_confidence_window_size=${win} \
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

# 2) Default decoding with fast inference
for model_name in "${MODEL_NAMES[@]}"; do
  for tmp in "${DEFAULT_TEMPS[@]}"; do
    name="$(build_run_name_default "$tmp")"
    cmd=(uv run python3 evaluate-decode.py \
      eval.run_default=true eval.run_custom=false \
      model.fast_inference=true \
      model.model_name="${model_name}" \
      eval.batch_size_groups=${G} \
      eval.num_generations=${N} \
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
