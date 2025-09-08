#!/usr/bin/env bash
# Sweep SMC decoding hyperparameters with evaluate-decode.py
#
# Usage examples (from repo root):
#   bash scripts/sweep_decode_smc.sh
#   WANDB=false REPEAT=1 bash scripts/sweep_decode_smc.sh
#   MODEL=Qwen/Qwen2.5-3B DATASETS="amc23,gsm8k" bash scripts/sweep_decode_smc.sh
#   GROUPS=8 NUM_GEN=8 PREFIX=my-sweep bash scripts/sweep_decode_smc.sh
#
# Env vars (optional):
#   ESS_VALUES   : space-separated list of ESS thresholds (default: "0.1 0.2 0.3 0.4 0.5")
#   WIN_VALUES   : space-separated list of window sizes (default: "25 50 75 100 125 150 175 200")
#   RUN_DEFAULT  : run default decoding (true/false, default: false)
#   RUN_CUSTOM   : run custom SMC decoding (true/false, default: true)
#   WANDB        : enable W&B logging (true/false, default: true)
#   REPEAT       : eval.repeat_cnt override (default: leave config value)
#   PREFIX       : W&B run name prefix (default: "decode-smc")
#   MODEL        : model.model_name override (e.g., Qwen/Qwen2.5-3B)
#   DATASETS     : comma-separated datasets (e.g., "amc23,gsm8k")
#   BATCH        : eval.batch_size override for default decoding
#   GROUPS       : eval.batch_size_groups override for custom decoding
#   NUM_GEN      : custom_decode.num_generations override

set -euo pipefail

ESS_VALUES=${ESS_VALUES:-"0.1 0.2 0.3 0.4 0.5"}
WIN_VALUES=${WIN_VALUES:-"25 50 75 100 125 150 175 200"}
RUN_DEFAULT=${RUN_DEFAULT:-false}
RUN_CUSTOM=${RUN_CUSTOM:-true}
WANDB=${WANDB:-true}

# Derive default prefix as smc-{model_name} if not provided.
# MODEL env overrides config; falls back to config/decode_eval.yaml.
CONFIG_FILE=${CONFIG_FILE:-config/decode_eval.yaml}
if [ -z "${PREFIX:-}" ]; then
  model_src="${MODEL:-}"
  if [ -z "$model_src" ] && [ -f "$CONFIG_FILE" ]; then
    # Try to parse model_name from YAML (simple grep/sed, handles quoted or unquoted)
    model_src=$(grep -E "^[[:space:]]*model_name:" "$CONFIG_FILE" | head -n 1 | sed -E 's/.*model_name:[[:space:]]*"?([^"#]+)"?.*/\1/')
  fi
  if [ -z "$model_src" ]; then
    model_src="unknown-model"
  fi
  model_sanitized="${model_src//\//-}"
  model_sanitized="${model_sanitized// /-}"
  PREFIX="smc-${model_sanitized}"
fi

# GPU scheduler: distribute jobs across visible GPUs (one per GPU)
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPUS=${#GPUS[@]}
declare -a PIDS

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
      echo $idx; return 0
    fi
  done
  echo -1
}

jobs_submitted=0
for ess in ${ESS_VALUES}; do
  for w in ${WIN_VALUES}; do
    name="${PREFIX}-ess${ess}-win${w}"
    cmd=(uv run python3 evaluate-decode.py \
      eval.run_default=${RUN_DEFAULT} \
      eval.run_custom=${RUN_CUSTOM} \
      custom_decode.smc_ess_threshold=${ess} \
      custom_decode.smc_confidence_window_size=${w} \
      wandb.run_name=${name})

    if [ "${WANDB}" != "true" ]; then
      cmd+=("wandb.enable=false")
    fi
    if [ -n "${MODEL:-}" ]; then
      cmd+=("model.model_name=${MODEL}")
    fi
    if [ -n "${DATASETS:-}" ]; then
      cmd+=("datasets=[${DATASETS}]")
    fi
    if [ -n "${REPEAT:-}" ]; then
      cmd+=("eval.repeat_cnt=${REPEAT}")
    fi
    if [ -n "${BATCH:-}" ]; then
      cmd+=("eval.batch_size=${BATCH}")
    fi
    if [ -n "${GROUPS:-}" ]; then
      cmd+=("eval.batch_size_groups=${GROUPS}")
    fi
    if [ -n "${NUM_GEN:-}" ]; then
      cmd+=("custom_decode.num_generations=${NUM_GEN}")
    fi

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

# Wait for all running jobs
for pid in "${PIDS[@]}"; do
  if [ -n "$pid" ]; then
    wait "$pid"
  fi
done

echo "Completed ${jobs_submitted} jobs."
