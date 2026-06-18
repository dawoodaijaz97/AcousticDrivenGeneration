#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_b20_pd2x
#SBATCH --output=runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/eval_slurm_%j.out
#SBATCH --error=runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/eval_slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$WORK/AcousticDrivenGeneration"
mkdir -p runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x

echo "=== B20 (PD 2x oversample + LoRA r=32 on B17) decode eval ==="
python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/final_model \
  --tokenizer-model "$WORK/models/flan-t5-base" \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/test_decode_metrics.json \
  --output-predictions-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/test_decode_predictions.json \
  --batch-size 8 \
  --seed 42 \
  --require-gpu

echo "Done. Metrics file:"
ls -la runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/test_decode_metrics.json
echo "=== AVG / per-metric ==="
python -c "import json; d=json.load(open('runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/test_decode_metrics.json')); o=d['overall']; print({k: round(o[k],4) for k in ['R_1','R_2','R_L','BLEU','BERT','AVG']}); bg=d.get('by_group',{}); print('PD', round(bg['PD']['AVG'],4), 'HC', round(bg['HC']['AVG'],4))"

echo "=== PD structure analysis ==="
python -m main.analyze_pd_decode \
  --predictions-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/test_decode_predictions.json \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/pd_analysis.json
