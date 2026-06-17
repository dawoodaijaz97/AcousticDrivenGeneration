#!/bin/bash -l
# B19 prepare — CPU on tinyx login (or woody). Re-tokenize required for new prompt style.
set -euo pipefail

cd "$WORK/AcousticDrivenGeneration"
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

python -m main.prepare \
  --output-dir data/processed/flan-t5-base/100k-flan-paper-report-template \
  --train-size 100k --tokenize \
  --tokenizer-model "$WORK/models/flan-t5-base" \
  --prompt-style flan-paper-report-template
