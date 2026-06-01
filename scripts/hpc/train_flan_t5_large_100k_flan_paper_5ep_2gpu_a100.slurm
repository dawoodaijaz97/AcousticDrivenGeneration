#!/bin/bash -l
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=flan_large_l5_5ep_2gpu
#SBATCH --output=runs/flan-t5-large/100k-flan-paper-5ep/slurm_%j.out
#SBATCH --error=runs/flan-t5-large/100k-flan-paper-5ep/slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs/flan-t5-large/100k-flan-paper-5ep
cd "$WORK/AcousticDrivenGeneration"

# 2x A100 via torchrun DDP. Effective batch = per_device(2) * nproc(2) * grad_accum(1) = 4,
# matching B11 / the single-GPU large runs so the epochs/size comparison stays clean.
torchrun --standalone --nproc_per_node=2 -m main.train \
  --tokenized-dir data/processed/flan-t5-large/100k-flan-paper/tokenized \
  --output-dir runs/flan-t5-large/100k-flan-paper-5ep \
  --model-name "$WORK/models/flan-t5-large" \
  --num-train-epochs 5 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --logging-steps 100 \
  --eval-steps 5000 \
  --save-steps 5000 \
  --save-total-limit 2 \
  --seed 42 \
  --require-gpu
