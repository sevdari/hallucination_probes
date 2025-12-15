#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --job-name=paper_reproduce
#SBATCH --output=log/%x-%j_%a.out
#SBATCH --error=log/%x-%j_%a.err
#SBATCH --array=0-1
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail


export LOCAL_PROBES_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/probes
export TRANSFORMERS_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache

# private keys
export HF_TOKEN="$(cat ~/keys/.hf_token)"
export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token_write)"
if [[ -f ~/keys/.wandb_key ]]; then
    export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
fi

# change this to your repo path
cd /workspace/hallucination_probes/

git pull
git checkout  compare_models
git pull



export CUDA_VISIBLE_DEVICES=0
uv pip install --no-deps  . --system

# multiarray job setup
CONFIGS=(train_config_apertus_no_lora_our_dataset.yaml train_config_apertus_our_dataset.yaml)
SELECTED_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running with config: $SELECTED_CONFIG"

python probe/train.py --config=configs/$SELECTED_CONFIG

# srun --account=large-sc-2 -p normal  --time=01:00:00 --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml --pty bash
