#!/bin/bash -l
# Submit optional LoRA rank sweep (B16 r=8, B17 r=32) after syncing code to cluster.
#
# Usage on tinyx:
#   cd $WORK/AcousticDrivenGeneration
#   sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_b16_lora8_a100.slurm
#   sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_b17_lora32_a100.slurm
#   sed -i 's/\r$//' scripts/hpc/eval_flan_t5_base_b16_lora8_a100.slurm
#   sed -i 's/\r$//' scripts/hpc/eval_flan_t5_base_b17_lora32_a100.slurm
#   bash scripts/hpc/submit_lora_rank_sweep.sh

set -euo pipefail
cd "${WORK:?}/AcousticDrivenGeneration"

for f in \
  scripts/hpc/train_flan_t5_base_b16_lora8_a100.slurm \
  scripts/hpc/train_flan_t5_base_b17_lora32_a100.slurm \
  scripts/hpc/eval_flan_t5_base_b16_lora8_a100.slurm \
  scripts/hpc/eval_flan_t5_base_b17_lora32_a100.slurm
do
  sed -i 's/\r$//' "$f"
done

B16=$(sbatch.tinygpu --parsable scripts/hpc/train_flan_t5_base_b16_lora8_a100.slurm)
B17=$(sbatch.tinygpu --parsable scripts/hpc/train_flan_t5_base_b17_lora32_a100.slurm)
echo "Submitted B16 train job $B16"
echo "Submitted B17 train job $B17"

E16=$(sbatch.tinygpu --parsable --dependency=afterok:"$B16" scripts/hpc/eval_flan_t5_base_b16_lora8_a100.slurm)
E17=$(sbatch.tinygpu --parsable --dependency=afterok:"$B17" scripts/hpc/eval_flan_t5_base_b17_lora32_a100.slurm)
echo "Submitted B16 eval job $E16 (after $B16)"
echo "Submitted B17 eval job $E17 (after $B17)"

echo "Queue: squeue.tinygpu -u $USER"
