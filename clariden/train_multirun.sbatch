#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --job-name=even-layer-seeds
#SBATCH --output=log/%x-%j_%a.out
#SBATCH --error=log/%x-%j_%a.err
# 48 total runs / 4 GPUs per node = 12 array tasks (0-11)
#SBATCH --array=0-11
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail

export LOCAL_PROBES_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/probes
export TRANSFORMERS_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache

# private keys
export HF_TOKEN="$(cat ~/keys/.hf_token)"
if [[ -f ~/keys/.wandb_key ]]; then
    export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
fi

if [[ -f ~/keys/.hf_token_write ]]; then
    export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"
fi

cd /workspace/hallucination_probes/

git pull
git checkout transferability
git pull

uv pip install --no-deps . --system
uv pip install hydra-core --system

mkdir -p log

# Define search space
MODELS=(llama apertus)
EVEN_LAYERS=(4 10 16 20 24 26 28 30)
SEEDS=(42 43 44)

# Build a flat list of all combinations
RUN_MODELS=()
RUN_LAYERS=()
RUN_SEEDS=()

for m in "${MODELS[@]}"; do
  for l in "${EVEN_LAYERS[@]}"; do
    for s in "${SEEDS[@]}"; do
      RUN_MODELS+=("$m")
      RUN_LAYERS+=("$l")
      RUN_SEEDS+=("$s")
    done
  done
done

TOTAL=${#RUN_MODELS[@]}          # Will be 8 * 3 * 2 = 48
TASKS_PER_JOB=4                  # 4 GPUs per SLURM job
START=$((SLURM_ARRAY_TASK_ID * TASKS_PER_JOB))

# Launch exactly 4 tasks in the background, mapping to GPU 0, 1, 2, 3
for local_gpu in 0 1 2 3; do
  idx=$((START + local_gpu))
  
  # Safety check just in case TOTAL isn't perfectly divisible by 4
  if [[ $idx -ge $TOTAL ]]; then
    break
  fi

  MODEL="${RUN_MODELS[$idx]}"
  LAYER="${RUN_LAYERS[$idx]}"
  SEED="${RUN_SEEDS[$idx]}"
  
  echo "Launching Run $idx/$TOTAL -> Model: $MODEL | Layer: $LAYER | Seed: $SEED on GPU: $local_gpu"

  # Override probe_id on the fly so runs don't overwrite each other in output dirs / wandb
  CUDA_VISIBLE_DEVICES=$local_gpu python probe/train.py \
    model=$MODEL \
    training=no_lora \
    dataset=our_long_form \
    probe_config.layer=$LAYER \
    seed=$SEED \
    "probe_config.probe_id=\${model.short_name}_no_lora_long_form_layer${LAYER}_seed${SEED}" &
done

# Wait for all 4 background tasks on this node to finish
wait
echo "All 4 runs for Array ID $SLURM_ARRAY_TASK_ID completed!"
