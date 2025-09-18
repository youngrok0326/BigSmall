#!/usr/bin/env bash
# CDF-specific SMC sweep with dynamic GPU scheduling

set -euo pipefail

MODEL_NAMES=(${MODEL_NAMES:-"Qwen/Qwen2.5-1.5B-Instruct"})
CDF_ALPHAS=(0.5 0.4)
SMC_TOPKS=(20 40)
CONF_GROUPS=(mean geo)
AGGREGATIONS=(mean prod min last)
WINDOW_SIZES=(1 8 32 64 128 256 512)
BATCH_GEN_PAIRS=(
  "128 8"
)

DEFAULT_TEMPS=()

# GPU scheduler (mirrors sweep_decode_selected.sh)
detect_visible_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    local IFS=','
    read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    if [ ${#GPUS[@]} -gt 0 ]; then return; fi
  fi
  local count
  count=$(python3 - <<'PY' 2>/dev/null || echo 0
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

sanitize_model_name() { echo -n "$1" | sed 's#/#-#g'; }

build_run_name_cdf() {
  local model_name="$1"; local alpha="$2"; local topk="$3"; local group="$4"; local agg="$5"; local win="$6"; local groups="$7"; local ngen="$8"
  local safe_model
  safe_model=$(sanitize_model_name "$model_name")
  echo "smc_${safe_model}_score-cdf_alpha${alpha}_topk${topk}_group-${group}_agg-${agg}_win${win}_G${groups}_N${ngen}"
}

jobs_submitted=0

for pair in "${BATCH_GEN_PAIRS[@]}"; do
  read -r batch_groups num_gen <<< "$pair"
  for model_name in "${MODEL_NAMES[@]}"; do
    for alpha in "${CDF_ALPHAS[@]}"; do
      for topk in "${SMC_TOPKS[@]}"; do
        for group in "${CONF_GROUPS[@]}"; do
          for agg in "${AGGREGATIONS[@]}"; do
            for win in "${WINDOW_SIZES[@]}"; do
              name="$(build_run_name_cdf "$model_name" "$alpha" "$topk" "$group" "$agg" "$win" "$batch_groups" "$num_gen")"
              cmd=(uv run python3 evaluate-decode.py \
                eval.run_default=false eval.run_custom=true \
                model.model_name="${model_name}" \
                eval.batch_size_groups=${batch_groups} \
                eval.num_generations=${num_gen} \
                custom_decode.smc_topk=${topk} \
                custom_decode.smc_confidence_window_size=${win} \
                custom_decode.confidence.scoring=cdf \
                custom_decode.confidence.cdf_alpha=${alpha} \
                custom_decode.confidence.group=${group} \
                custom_decode.confidence.aggregation=${agg} \
                wandb.run_name=${name})

              while true; do
                slot=$(find_free_gpu)
                if [ "$slot" -ge 0 ]; then
                  echo "Running (CDF): ${cmd[*]}"
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
done

for pid in "${PIDS[@]}"; do
  if [ -n "${pid:-}" ]; then
    wait "$pid"
  fi
done

echo "Completed ${jobs_submitted} jobs."
