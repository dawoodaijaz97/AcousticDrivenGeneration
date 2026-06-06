#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_b12_wd0
#SBATCH --output=runs/flan-t5-base/100k-flan-paper-5ep-wd0/eval_slurm_%j.out
#SBATCH --error=runs/flan-t5-base/100k-flan-paper-5ep-wd0/eval_slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$WORK/AcousticDrivenGeneration"
mkdir -p runs/flan-t5-base/100k-flan-paper-5ep-wd0

echo "=== B12 (base 100k-flan-paper-5ep-wd0) decode eval ==="
python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-wd0/final_model \
  --tokenizer-model "$WORK/models/flan-t5-base" \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-wd0/test_decode_metrics.json \
  --batch-size 8 \
  --seed 42

echo "Done. Metrics file:"
ls -la runs/flan-t5-base/100k-flan-paper-5ep-wd0/test_decode_metrics.json
echo "=== AVG / per-metric ==="
python -c "import json; d=json.load(open('runs/flan-t5-base/100k-flan-paper-5ep-wd0/test_decode_metrics.json'))['overall']; print({k: round(d[k],4) for k in ['R_1','R_2','R_L','BLEU','BERT','AVG']})"
