#!/bin/bash
# Interactive eval on tinyx (after: conda activate acoustic, module load python).
# Usage: bash scripts/hpc/eval_flan_t5_small_s4_s5.sh

set -euo pipefail
cd "${WORK:-$HOME}/AcousticDrivenGeneration"

export HF_HOME="${HF_HOME:-$WORK/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TOKENIZED=data/processed/flan-t5-small/100k/tokenized
TOKENIZER=google/flan-t5-small

python -m main.eval_decode \
  --tokenized-dir "$TOKENIZED" \
  --model-path runs/flan-t5-small/100k-lr5e4/final_model \
  --tokenizer-model "$TOKENIZER" \
  --output-json runs/flan-t5-small/100k-lr5e4/test_decode_metrics.json \
  --batch-size 8 \
  --seed 42

python -m main.eval_decode \
  --tokenized-dir "$TOKENIZED" \
  --model-path runs/flan-t5-small/100k-lr3e4/final_model \
  --tokenizer-model "$TOKENIZER" \
  --output-json runs/flan-t5-small/100k-lr3e4/test_decode_metrics.json \
  --batch-size 8 \
  --seed 42

echo "Wrote:"
echo "  runs/flan-t5-small/100k-lr5e4/test_decode_metrics.json"
echo "  runs/flan-t5-small/100k-lr3e4/test_decode_metrics.json"
