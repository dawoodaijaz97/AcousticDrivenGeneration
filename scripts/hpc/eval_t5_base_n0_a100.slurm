#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_n0_t5_base
#SBATCH --output=runs/t5-base/100k-flan-paper-5ep/eval_slurm_%j.out
#SBATCH --error=runs/t5-base/100k-flan-paper-5ep/eval_slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$WORK/AcousticDrivenGeneration"
mkdir -p runs/t5-base/100k-flan-paper-5ep

echo "=== N0 (t5-base 100k-flan-paper-5ep) decode eval ==="
python -m main.eval_decode \
  --tokenized-dir data/processed/t5-base/100k-flan-paper/tokenized \
  --model-path runs/t5-base/100k-flan-paper-5ep/final_model \
  --tokenizer-model "$WORK/models/t5-base" \
  --output-json runs/t5-base/100k-flan-paper-5ep/test_decode_metrics.json \
  --batch-size 8 \
  --seed 42 \
  --require-gpu

echo "Done. Metrics file:"
ls -la runs/t5-base/100k-flan-paper-5ep/test_decode_metrics.json
echo "=== AVG / per-metric ==="
python -c "import json; d=json.load(open('runs/t5-base/100k-flan-paper-5ep/test_decode_metrics.json'))['overall']; print({k: round(d[k],4) for k in ['R_1','R_2','R_L','BLEU','BERT','AVG']})"
