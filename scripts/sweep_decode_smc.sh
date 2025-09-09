#!/usr/bin/env bash
# Sweep SMC decoding hyperparameters with evaluate-decode.py
#
# Usage:
#   bash scripts/sweep_decode_smc.sh
#   CUDA_VISIBLE_DEVICES=0,1 ESS_VALUES="0.2 0.4" WIN_VALUES="50 100" bash scripts/sweep_decode_smc.sh
# Notes:
#   - If CUDA_VISIBLE_DEVICES is unset, the script auto-detects all GPUs via nvidia-smi/torch
#   - One concurrent job per GPU; next job is queued on the first GPU that finishes
#
# Env vars:
#   ESS_VALUES : space-separated ESS thresholds (default shown below)
#   WIN_VALUES : space-separated window sizes (default shown below)
#
# Everything else strictly follows config/decode_eval.yaml.

set -euo pipefail

ESS_VALUES=${ESS_VALUES:-"0.1 0.2 0.3 0.4 0.5"}
WIN_VALUES=${WIN_VALUES:-"25 50 75 100 125 150 175 200 225 250"}

# Model and group configs to sweep
MODEL_NAMES=(
  "Qwen/Qwen2.5-3B"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-Math-1.5B-Instruct"
)
declare -a GEN_GROUPS
# (batch_size_groups, num_generations)
GEN_GROUPS+=("32 16")
GEN_GROUPS+=("8 64")

# Build run name: smc_{model_name}_num_gen{num_generation}_ess{ess}_win{win}
# Sanitize model name for filenames (replace '/' with '-')
build_run_name() {
  local model_name="$1"; local num_gen="$2"; local ess="$3"; local win="$4"
  local safe_model_name
  safe_model_name=$(echo -n "$model_name" | sed 's#/#-#g')
  echo "smc_${safe_model_name}_num_gen${num_gen}_ess${ess}_win${win}"
}

# GPU scheduler: distribute jobs across visible GPUs (one per GPU)
detect_visible_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Read one GPU index per line. Do NOT strip newlines, or indices will concatenate (e.g. "01234567").
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    if [ ${#GPUS[@]} -gt 0 ]; then return; fi
  fi
  # Fallback to torch if available
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
  local gpu_index=$1
  shift
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

jobs_submitted=0
for model_name in "${MODEL_NAMES[@]}"; do
  for grp in "${GEN_GROUPS[@]}"; do
    IFS=' ' read -r G N <<< "$grp"
    if [ -z "${G:-}" ] || [ -z "${N:-}" ]; then
      echo "Invalid GEN_GROUPS entry: '$grp' (expected 'G N')" >&2
      exit 1
    fi
    for ess in ${ESS_VALUES}; do
      for w in ${WIN_VALUES}; do
        name="$(build_run_name "$model_name" "$N" "$ess" "$w")"
        cmd=(uv run python3 evaluate-decode.py \
          eval.run_default=false eval.run_custom=true \
          model.model_name="${model_name}" \
          eval.batch_size_groups=${G} \
          custom_decode.num_generations=${N} \
          custom_decode.smc_ess_threshold=${ess} \
          custom_decode.smc_confidence_window_size=${w} \
          wandb.run_name=${name})

        while true; do
          slot=$(find_free_gpu)
          if [ "$slot" -ge 0 ]; then
            echo "Running: ${cmd[*]}"
            submit_job "$slot" "${cmd[@]}"
            jobs_submitted=$((jobs_submitted+1))
            break
          fi
          # No free GPU; wait a bit and retry
          sleep 2
        done
      done
    done
  done
done

# Wait for all running jobs
for pid in "${PIDS[@]}"; do
  if [ -n "$pid" ]; then
    wait "$pid"
  fi
done

echo "Completed ${jobs_submitted} jobs."
