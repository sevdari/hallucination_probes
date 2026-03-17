#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p normal
#SBATCH --time=3:00:00
#SBATCH --job-name=probe-seed-ablation
#SBATCH --output=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/%x-%j_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/%x-%j_%a.err
#SBATCH --array=0-11
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail

export LOCAL_PROBES_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/probes
export TRANSFORMERS_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache

export HF_TOKEN="$(cat ~/keys/.hf_token)"
if [[ -f ~/keys/.wandb_key ]]; then
    export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
fi
if [[ -f ~/keys/.hf_token_write ]]; then
    export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"
fi

cd /workspace/hallucination_probes/
git pull
git checkout improve_stability
git pull

uv pip install --no-deps . --system
uv pip install hydra-core --system

mkdir -p log

# Fixed hyperparameters
MODEL="apertus"
LR="3e-4"
PROBE_DTYPE="float32"

# 2x2 grid: {noLN, LN} x {noLoRA, LoRA} — all fp32
# Naming: <lora>_<norm>
VARIANTS=(fp32_no_ln fp32_ln lora_no_ln lora_ln)
EVEN_LAYERS=(10 20 26 30)
SEEDS=(42 43 44)

# Build flat list: variants -> layers -> seeds
RUN_VARIANTS=()
RUN_LAYERS=()
RUN_SEEDS=()

for v in "${VARIANTS[@]}"; do
  for l in "${EVEN_LAYERS[@]}"; do
    for s in "${SEEDS[@]}"; do
      RUN_VARIANTS+=("$v")
      RUN_LAYERS+=("$l")
      RUN_SEEDS+=("$s")
    done
  done
done

TOTAL=${#RUN_LAYERS[@]}   # 4 variants * 4 layers * 3 seeds = 48
TASKS_PER_JOB=4
START=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB))

for local_gpu in 0 1 2 3; do
  idx=$((START + local_gpu))

  if [[ $idx -ge $TOTAL ]]; then
    break
  fi

  VARIANT="${RUN_VARIANTS[$idx]}"
  LAYER="${RUN_LAYERS[$idx]}"
  SEED="${RUN_SEEDS[$idx]}"

  # Decode variant flags
  if [[ "$VARIANT" == *_ln ]]; then
    NORM="layernorm"
  else
    NORM="none"
  fi

  if [[ "$VARIANT" == lora_* ]]; then
    LORA_LAYERS="all"
  else
    LORA_LAYERS="none"
  fi

  PROBE_ID="apertus_seed_ablation_${VARIANT}_layer${LAYER}_seed${SEED}"

  echo "Run $idx/$TOTAL -> Variant: $VARIANT | Layer: $LAYER | Seed: $SEED | GPU: $local_gpu"

  CUDA_VISIBLE_DEVICES=$local_gpu python probe/train.py \
    model=$MODEL \
    training=no_lora \
    dataset=our_long_form \
    probe_config.layer=$LAYER \
    model_dtype=bfloat16 \
    probe_config.probe_dtype=$PROBE_DTYPE \
    probe_config.normalize_before_head=$NORM \
    probe_config.lora_layers=$LORA_LAYERS \
    probe_head_lr=$LR \
    seed=$SEED \
    "probe_config.probe_id=${PROBE_ID}" &
done

wait
echo "All tasks for Array ID $SLURM_ARRAY_TASK_ID completed."