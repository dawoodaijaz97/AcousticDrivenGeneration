#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_b23_lora32
#SBATCH --output=runs/flan-t5-base/100k-structured-seven-label/eval_slurm_%j.out
#SBATCH --error=runs/flan-t5-base/100k-structured-seven-label/eval_slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$WORK/AcousticDrivenGeneration"
mkdir -p runs/flan-t5-base/100k-structured-seven-label

python -m main.eval_structured \
  --tokenized-dir data/processed/flan-t5-base/100k-structured-seven-label/tokenized \
  --model-path runs/flan-t5-base/100k-structured-seven-label/final_model \
  --tokenizer-model "$WORK/models/flan-t5-base" \
  --output-json runs/flan-t5-base/100k-structured-seven-label/test_decode_metrics.json \
  --output-predictions-json runs/flan-t5-base/100k-structured-seven-label/test_decode_predictions.json \
  --batch-size 8 \
  --max-new-tokens 128 \
  --seed 42 \
  --require-gpu

echo "=== B23 label + verbalized metrics ==="
python -c "import json; d=json.load(open('runs/flan-t5-base/100k-structured-seven-label/test_decode_metrics.json')); print('label exact_match', d['label_metrics']['exact_match_rate']); print('label coverage', d['label_structure_metrics']['category_coverage']); print('verbalized AVG', round(d['verbalized_overall']['AVG'],4))"
