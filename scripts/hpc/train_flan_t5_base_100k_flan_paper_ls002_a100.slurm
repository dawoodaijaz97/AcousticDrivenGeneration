#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=flan_base_b10_ls002
#SBATCH --output=runs/flan-t5-base/100k-flan-paper-ls002/slurm_%j.out
#SBATCH --error=runs/flan-t5-base/100k-flan-paper-ls002/slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs/flan-t5-base/100k-flan-paper-ls002
cd "$WORK/AcousticDrivenGeneration"

python -m main.train \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --output-dir runs/flan-t5-base/100k-flan-paper-ls002 \
  --model-name "$WORK/models/flan-t5-base" \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --learning-rate 5e-4 \
  --weight-decay 0.01 \
  --label-smoothing 0.02 \
  --warmup-ratio 0.1 \
  --logging-steps 100 \
  --eval-steps 5000 \
  --save-steps 5000 \
  --save-total-limit 2 \
  --seed 42 \
  --require-gpu
