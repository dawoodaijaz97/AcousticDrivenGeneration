#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_s4_s5_small
#SBATCH --output=runs/flan-t5-small/eval_s4_s5_slurm_%j.out
#SBATCH --error=runs/flan-t5-small/eval_s4_s5_slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$WORK/AcousticDrivenGeneration"
mkdir -p runs/flan-t5-small

TOKENIZED=data/processed/flan-t5-small/100k/tokenized
TOKENIZER=google/flan-t5-small

echo "=== S4 (100k-lr5e4) ==="
python -m main.eval_decode \
  --tokenized-dir "$TOKENIZED" \
  --model-path runs/flan-t5-small/100k-lr5e4/final_model \
  --tokenizer-model "$TOKENIZER" \
  --output-json runs/flan-t5-small/100k-lr5e4/test_decode_metrics.json \
  --batch-size 8 \
  --seed 42

echo "=== S5 (100k-lr3e4) ==="
python -m main.eval_decode \
  --tokenized-dir "$TOKENIZED" \
  --model-path runs/flan-t5-small/100k-lr3e4/final_model \
  --tokenizer-model "$TOKENIZER" \
  --output-json runs/flan-t5-small/100k-lr3e4/test_decode_metrics.json \
  --batch-size 8 \
  --seed 42

echo "Done. Metrics:"
ls -la runs/flan-t5-small/100k-lr5e4/test_decode_metrics.json \
       runs/flan-t5-small/100k-lr3e4/test_decode_metrics.json
