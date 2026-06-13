#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=02:00:00
#SBATCH --job-name=analyze_b14_pd
#SBATCH --output=runs/flan-t5-base/100k-flan-paper-5ep-lora16/pd_analysis_slurm_%j.out
#SBATCH --error=runs/flan-t5-base/100k-flan-paper-5ep-lora16/pd_analysis_slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN=runs/flan-t5-base/100k-flan-paper-5ep-lora16
BASELINE=runs/flan-t5-base/100k-flan-paper-5ep

cd "$WORK/AcousticDrivenGeneration"
mkdir -p "$RUN"

echo "=== B14 decode + predictions dump ==="
python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path "$RUN/final_model" \
  --tokenizer-model "$WORK/models/flan-t5-base" \
  --output-json "$RUN/test_decode_metrics.json" \
  --output-predictions-json "$RUN/test_decode_predictions.json" \
  --batch-size 8 \
  --seed 42 \
  --require-gpu

echo "=== B11 baseline predictions (for comparison) ==="
python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path "$BASELINE/final_model" \
  --tokenizer-model "$WORK/models/flan-t5-base" \
  --output-predictions-json "$BASELINE/test_decode_predictions_b14_analysis.json" \
  --batch-size 8 \
  --seed 42 \
  --require-gpu

echo "=== PD / per-category analysis ==="
python -m main.analyze_pd_decode \
  --predictions-json "$RUN/test_decode_predictions.json" \
  --baseline-predictions-json "$BASELINE/test_decode_predictions_b14_analysis.json" \
  --output-json "$RUN/pd_analysis.json"

echo "=== B14 PD vs HC (decode metrics) ==="
python -c "
import json
d = json.load(open('$RUN/test_decode_metrics.json'))
for g in ('PD','HC'):
    b = d['by_group'][g]
    print(g, {k: round(b[k],4) for k in ('AVG','BLEU','R_1','R_2')})
"

echo "=== PD category gaps (B14 vs B11) ==="
python -c "
import json
d = json.load(open('$RUN/pd_analysis.json'))
print('baseline_deltas', d.get('baseline_deltas'))
a = d['analysis']
print('B14 PD', a['by_group'].get('PD'))
print('B14 HC', a['by_group'].get('HC'))
"

echo "Done. Artifacts:"
ls -la "$RUN/test_decode_predictions.json" "$RUN/pd_analysis.json"
