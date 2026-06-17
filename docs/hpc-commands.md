# HPC commands (NHR@FAU — TinyGPU)

Quick reference for `**tinyx**` (`tinyx.nhr.fau.de`). Slurm commands use the `**.tinygpu**` suffix. Project path: `**$WORK/AcousticDrivenGeneration**`.

SSH from Windows: `ssh tinyx` (see `~/.ssh/config` — jump via `nhr-jump`). More detail: [HPC Info.pdf](HPC%20Info.pdf), `.cursor/rules/hpc-nhr-fau.mdc`.

---

## Login and environment

```bash
ssh tinyx
cd $WORK/AcousticDrivenGeneration

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic
```

**Hugging Face (login node — downloads only):**

```bash
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME=$WORK/huggingface
```

**Offline on compute / after models are cached:**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Use `**$WORK/models/flan-t5-{small,base,large}**` for `--model-name` / `--tokenizer-model` on compute nodes (no Hub access).

---

## Check GPU allocation and queue

```bash
# Your jobs (RUNNING, PENDING, etc.)
squeue.tinygpu -u iwi5439h

# More detail (partition, time, node, GPUs)
squeue.tinygpu -u iwi5439h -o "%.18i %.9P %.14j %.8u %.2t %.10M %.6D %R %b"

# Cluster / partition status (alloc = full, mix = partly free)
sinfo.tinygpu

# One job after submit
scontrol.tinygpu show job JOBID

# Finished job accounting
sacct.tinygpu -j JOBID --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS

# Storage quota
shownicerquota.pl
```

---

## Interactive GPU session

Request a GPU; wait in `**PD**` if the cluster is busy. When you get a shell:

```bash
# A100 (40 GB) — training / base eval
salloc.tinygpu --gres=gpu:a100:1 --partition=a100 --time=02:00:00

# V100 (32 GB) — training / eval; use FP32 (no --fp16 on V100)
salloc.tinygpu --gres=gpu:v100:1 --partition=v100 --time=02:00:00

# RTX 3080 (10 GB) — small-model eval if A100/V100 are queued
salloc.tinygpu --gres=gpu:rtx3080:1 --partition=rtx3080 --time=01:00:00
```

**Confirm you are on a compute node (not the login node):**

```bash
hostname          # should be tg0xx, not tinyx
nvidia-smi

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate acoustic
cd $WORK/AcousticDrivenGeneration
```

**Leave the session:** `exit`

**Cancel a queued/running job:** `scancel.tinygpu JOBID`

---

## Batch training (Slurm)

Only **training** uses Slurm in this repo; **prepare** and **eval_decode** run on the login node or an interactive GPU session.

```bash
cd $WORK/AcousticDrivenGeneration

# Fix Windows line endings after scp
sed -i 's/\r$//' scripts/hpc/YOUR_JOB.slurm

sbatch.tinygpu scripts/hpc/YOUR_JOB.slurm
# → note job id, e.g. Submitted batch job 1645191

tail -f runs/<model>/<run>/slurm_JOBID.out
```

Examples: `scripts/hpc/train_flan_t5_base_100k_flan_paper_a100.slurm` (B5), `train_flan_t5_small_100k_lr5e4_a100.slurm` (S4), `prepare_flan_t5_base_b19_report_template.sh` + `train_flan_t5_base_b19_report_template_lora32_a100.slurm` (B19).

**Prepare and decode eval** (login or interactive GPU): see [data-pipeline.md](data-pipeline.md), [train.md](train.md), [eval-decode.md](eval-decode.md), [Model Improvement Plan.md](Model%20Improvement%20Plan.md).

---

## Sync metric JSONs (git — not scp)

Only `**test_decode_metrics.json`** and `**test_eval.json`** under `runs/` are tracked (see `.gitignore`). Checkpoints, `final_model/`, and Slurm logs stay local on HPC.

**On HPC after eval / train:**

```bash
cd $WORK/AcousticDrivenGeneration

# 1) Must have the updated .gitignore (exceptions for test_*.json)
git pull

# 2) Confirm files exist
ls -la runs/flan-t5-base/100k-flan-paper/test_*.json

# 3) Add explicitly (glob may not expand if no matches)
git add -f \
  runs/flan-t5-base/100k-flan-paper/test_decode_metrics.json \
  runs/flan-t5-base/100k-flan-paper/test_eval.json

git status
git commit -m "Add B5 eval metrics"
git push
```

If `git status` still shows nothing: old `.gitignore` without the `!/runs/**/test_*.json` lines — run `git pull` on HPC first, or copy `.gitignore` from your laptop.

**On Windows / laptop:**

```bash
cd ~/PycharmProjects/AcousticDrivenGeneration
git pull
```

First time adding metrics from an ignored tree, if `git add` skips files:

```bash
git add -f runs/<model>/<run>/test_decode_metrics.json runs/<model>/<run>/test_eval.json
```

---

## Copy code to cluster (no git)

```powershell
scp -r main etl docs requirements.txt scripts tinyx:/home/woody/iwi5/iwi5439h/AcousticDrivenGeneration/
```

---

## Related docs

- [data-pipeline.md](data-pipeline.md) — `main.prepare`
- [train.md](train.md) — `main.train`
- [eval-decode.md](eval-decode.md) — `main.eval_decode`
- [Model Improvement Plan.md](Model%20Improvement%20Plan.md) — experiment checklist
- [training_progress.md](training_progress.md) — results log

