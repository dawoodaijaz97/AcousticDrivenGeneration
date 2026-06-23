#!/bin/bash -l
#SBATCH --partition=woody
#SBATCH --time=04:00:00
#SBATCH --job-name=prep_b23
#SBATCH --output=runs/flan-t5-base/100k-structured-seven-label/prepare_slurm_%j.out
#SBATCH --error=runs/flan-t5-base/100k-structured-seven-label/prepare_slurm_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic

export HF_HOME=$WORK/huggingface
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

cd "$WORK/AcousticDrivenGeneration"
mkdir -p runs/flan-t5-base/100k-structured-seven-label

python -m main.prepare \
  --output-dir data/processed/flan-t5-base/100k-structured-seven-label \
  --train-size 100k \
  --tokenize \
  --tokenizer-model "$WORK/models/flan-t5-base" \
  --prompt-style flan-paper \
  --target-format structured-seven-label \
  --max-target-length 128

echo "Done. Tokenized data:"
ls -la data/processed/flan-t5-base/100k-structured-seven-label/tokenized
