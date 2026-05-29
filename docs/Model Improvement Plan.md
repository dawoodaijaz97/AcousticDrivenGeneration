# Model Improvement Plan

**What this file is:** the **plan** — what to try, in what order, and whether each task is done (`[x]` / `[ ]`).

**What it is not:** numeric results. Those live in **[training_progress.md](training_progress.md)** (summary tables, PD/HC breakdown, conclusions from finished runs).

### Workflow

1. Pick the next open task here (phases, checklist, experiment IDs).
2. Train / `eval_decode` → artifacts under `runs/<model>/<run>/`.
3. **Log metrics** in [training_progress.md](training_progress.md) (new table row + conclusions if useful).
4. **Mark `[x]`** on the matching task and experiment row in this file.

**Related:** [train.md](train.md), [eval-decode.md](eval-decode.md), [data-pipeline.md](data-pipeline.md), [paper-overview.md](paper-overview.md), [hpc-commands.md](hpc-commands.md) (queue, `salloc`, env, `scp`).

**Paper target (LLaMA-7B, ~Table 4 AVG):** ~**0.68**. Current best result: see **B4** in [training_progress.md](training_progress.md).

**Best small @ 100k:** **S4** (`5e-4`, AVG **0.437**) — see [training_progress.md](training_progress.md). **S5** (`3e-4` @ 100k) completed but **below S0**; no further small LR runs planned.

**Phase 2 in progress:** **B5** — flan-t5-base, 100k, **`flan-paper`** prompt, LR **5e-4** (same hyperparameters as **B4**). Artifacts: `data/processed/flan-t5-base/100k-flan-paper/`, `runs/flan-t5-base/100k-flan-paper/`.

---

## Master checklist

### Pipeline & evaluation discipline

- [x] ETL + canonical splits (`main/` data pipeline)
- [x] Tokenize + train loop (`main.prepare`, `main.train`)
- [x] Decode eval on real test (`main.eval_decode`, beam 3)
- [x] Log PD vs HC in decode JSON (`by_group`)
- [x] Use [training_progress.md](training_progress.md) as the results log (append rows after each eval)
- [ ] Compare **checkpoints** vs `final_model` (BLEU may differ from `eval_val_loss` best step)
- [ ] Optional: wandb / tensorboard (`--report-to`)
- [ ] Per-category keyword/slot checks on decoded text
- [ ] Wire `compute_metrics` + `--predict-with-generate` in `main/train.py` (medium priority)

### Model size — Flan-T5-small (~77M)

- [x] **S0** — baseline 100k, LR 3e-4, default prompt (`runs/flan-t5-small/100k`, AVG **0.362**)
- [x] **S1–S3** — 10k LR sweep (1e-4, 3e-4, 5e-4); S3 best AVG **0.459**
- [x] Slurm script for 100k @ 5e-4 (`scripts/hpc/train_flan_t5_small_100k_lr5e4_a100.slurm`)
- [x] Slurm script for 100k @ 3e-4 (`scripts/hpc/train_flan_t5_small_100k_lr3e4_a100.slurm`)
- [x] **S4** — 100k @ 5e-4 trained + `eval_decode` logged (**best small @ 100k**)
- [x] **S5** — 100k @ 3e-4 trained + `eval_decode` logged (worse than S0 — closed)
- [ ] Phase 2 prompt / `max_source_length` on **B4** (primary); optional small S4 only if time
- [ ] Phase 3 beam sweep on frozen small checkpoint
- [ ] Phase 4 LoRA / `google-t5/t5-small` comparison (if plateau)

### Model size — Flan-T5-base

- [x] **B0** — 100k, LR 3e-4 (AVG **0.395**)
- [x] **B4** — 100k, LR 5e-4 (AVG **0.522**)
- [ ] **B5** — Flan paper prefix (`--prompt-style flan-paper`); re-prepare + re-train with **B4** hyperparameters
- [ ] Phase 2 — `max_source_length` audit (128 / 256 / 384)
- [ ] Phase 3 — beam / length sweep on **B4** checkpoint (no retrain)
- [ ] Label smoothing / weight-decay ablations (if needed after prompt)

### Model size — Flan-T5-large

- [x] **L0** — 100k, LR 3e-4 (AVG **0.500**) — **reference large config**
- [x] **L4** — 100k, LR 5e-4 (AVG **0.500**, ≈ tie; use L0 for comparisons)
- [ ] Phase 2 prompt experiments (only if base Phase 2 wins)

### Cross-cutting (after LR / size baselines)

- [ ] Epochs / steps (3 vs 5) on best config per tier
- [ ] Batch + gradient accumulation sweep
- [ ] Warmup / weight decay / label smoothing trials
- [ ] `--no-eval-train` wall-clock trial (one 10k run)
- [ ] FP16 on A100 only (V100 had NaN with fp16)
- [ ] Leakage audit (`example_hash` train/val/test)

---

## Phased schedule

### Phase 0 — Baselines & logging

- [x] Small **S0** trained and decode-evaluated
- [x] Base **B0** / **B4** trained and decode-evaluated
- [x] Large **L0** / **L4** trained and decode-evaluated
- [x] **training_progress.md** updated with all decode runs above
- [ ] Optional: HPC replicate of S0 vs local S0
- [ ] Ensure every finished run has `test_decode_metrics.json` in its `runs/.../` folder

### Phase 1 — Training hyperparameters (LR & promotion)

- [x] Small: 10k LR sweep (**S1–S3**)
- [x] Base: 100k LR comparison (**B0** vs **B4** → use **5e-4**)
- [x] Large: 100k LR comparison (**L0** vs **L4** → use **3e-4**)
- [x] Small: **S4** / **S5** 100k complete — **use S4** (`5e-4`); see [training_progress.md](training_progress.md)
- [ ] Label smoothing + extra weight-decay trials
- [ ] `--no-eval-train` vs default (one 10k run)

### Phase 2 — Prompt and input (re-tokenize required)

- [ ] **B5** — Flan paper prefix `Generate a report for:` (code: `--prompt-style flan-paper`; vs default)
- [ ] Category hints in prefix (seven mFDA categories)
- [ ] Numeric formatting / category labels in feature string
- [ ] `max_source_length` — confirm no silent truncation
- [ ] Re-prepare → re-train **B4** recipe on best prompt (base first) — **B5** train: `scripts/hpc/train_flan_t5_base_100k_flan_paper_a100.slurm`

### Phase 3 — Decoding and checkpoints (cheap)

- [ ] Beam 1 vs 3 vs 5 on **B4** `final_model`
- [ ] Best `checkpoint-*` vs `final_model` vs best `eval_val_loss` step
- [ ] Document decode settings beside each `test_decode_metrics.json`

### Phase 4 — Efficiency (if quality plateaus)

- [ ] LoRA / adapters (`peft`, rank 8–32)
- [ ] Freeze encoder, train decoder
- [ ] Compare **google-t5/t5-small** (non-Flan) at same pipeline

---

## Workstream details

### A. Evaluation (do first)

| Task | Status |
|------|--------|
| Standard decode eval after each promising run | [x] |
| Log PD vs HC separately | [x] |
| Use decode metrics to select configs (not val loss alone) | [x] |
| Checkpoint comparison | [ ] |
| Training-time generation metrics | [ ] |

### B. Training hyperparameters

| Experiment | Small | Base | Large |
|------------|-------|------|-------|
| LR sweep | [x] 10k + 100k (S4/S5 logged) | [x] B0 vs B4 | [x] L0 vs L4 |
| Epochs / max-steps | [ ] | [ ] | [ ] |
| Batch + accum | [ ] | [ ] | [ ] |
| Warmup / weight decay | [ ] | [ ] | [ ] |
| Label smoothing | [ ] | [ ] | [ ] |

### C. Prompt and input

| Experiment | Status |
|------------|--------|
| Flan paper prefix (B5) | [ ] in flight |
| Category hints in prefix | [ ] |
| Numeric formatting | [ ] |
| `max_source_length` | [ ] |

### D. Data and splits

| Experiment | Status |
|------------|--------|
| Train size 10k vs 100k | [x] (both available) |
| Fast 1k smoke runs | [x] |
| `google-t5/t5-small` comparison | [ ] |
| `--no-eval-train` trial | [ ] |
| Leakage audit | [ ] |

### E. Model efficiency

| Action | Status |
|--------|--------|
| LoRA / adapters | [ ] |
| Freeze encoder | [ ] |
| Span-corruption pretrain | [ ] (out of scope unless plateau) |
| FP16 on A100 | [ ] |

### F. Decoding knobs

| Knob | Status |
|------|--------|
| Beam size sweep | [ ] (default beam 3 in eval) |
| `max_new_tokens` ~512 | [x] |
| `no_repeat_ngram` 2 | [x] (paper-aligned in eval) |

---

## Experiment log (plan tracker)

Run IDs link plan tasks to `runs/` folders. **Metrics:** [training_progress.md](training_progress.md) summary table.

| id | model | train | prompt | lr | done | output_dir | notes |
|----|-------|-------|--------|-----|------|------------|-------|
| **S0** | flan-t5-small | 100k | default | 3e-4 | [x] | `runs/flan-t5-small/100k` | Baseline |
| **S1** | flan-t5-small | 10k | default | 1e-4 | [x] | `runs/flan-t5-small/10k-lr1e4` | |
| **S2** | flan-t5-small | 10k | default | 3e-4 | [x] | `runs/flan-t5-small/10k-lr3e4` | Best HC BLEU at 10k |
| **S3** | flan-t5-small | 10k | default | 5e-4 | [x] | `runs/flan-t5-small/10k-lr5e4` | Promoted → S4 |
| **S4** | flan-t5-small | 100k | default | 5e-4 | [x] | `runs/flan-t5-small/100k-lr5e4` | **Best small @ 100k**; metrics in training_progress |
| **S5** | flan-t5-small | 100k | default | 3e-4 | [x] | `runs/flan-t5-small/100k-lr3e4` | Below S0; 10k→100k HC trend did not hold |
| **B0** | flan-t5-base | 100k | default | 3e-4 | [x] | `runs/flan-t5-base/100k` | |
| **B4** | flan-t5-base | 100k | default | 5e-4 | [x] | `runs/flan-t5-base/100k-lr5e4` | Best overall so far |
| **L0** | flan-t5-large | 100k | default | 3e-4 | [x] | `runs/flan-t5-large/100k` | Large reference |
| **L4** | flan-t5-large | 100k | default | 5e-4 | [x] | `runs/flan-t5-large/100k-lr5e4` | ≈ L0 |
| **B5** | flan-t5-base | 100k | flan-paper | 5e-4 | [ ] | `runs/flan-t5-base/100k-flan-paper` | Phase 2; beat B4 **0.522** |

---

## Success criteria

| Tier | Small | Base (primary push) |
|------|-------|---------------------|
| **Minimum** | Reproducible decode JSON | Same |
| **Target** | S4: AVG **> 0.362**, HC BLEU **> 0.117** | Phase 2 beat **B4** (0.522) |
| **Stretch** | Narrow gap to paper small-tier | Approach paper ~0.68 band |

---

## Next actions (immediate)

1. [ ] **B5 prepare** — interactive / login node (`python -m main.prepare`, see commands below).
2. [ ] **B5 train** — `sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_a100.slurm` (after tokenized data exists).
3. [ ] **B5 eval** — interactive GPU (`python -m main.eval_decode`, see below) → row in [training_progress.md](training_progress.md).
4. [ ] Plot refresh / optional beam sweep on **B4**.

---

## Commands cheat sheet

**B5 prepare (CPU — `tinyx` login; proxy for tokenizer if needed):**

```bash
cd $WORK/AcousticDrivenGeneration
conda activate acoustic
module load python
export HF_HOME=$WORK/huggingface
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

python -m main.prepare \
  --output-dir data/processed/flan-t5-base/100k-flan-paper \
  --train-size 100k --tokenize \
  --tokenizer-model $WORK/models/flan-t5-base \
  --prompt-style flan-paper
```

**B5 train (A100 — Slurm only):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_100k_flan_paper_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_a100.slurm
```

**B5 decode eval (GPU — interactive, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**Submit S4 (HPC):**

```bash
cd $WORK/AcousticDrivenGeneration
sed -i 's/\r$//' scripts/hpc/train_flan_t5_small_100k_lr5e4_a100.slurm   # if uploaded from Windows
sbatch.tinygpu scripts/hpc/train_flan_t5_small_100k_lr5e4_a100.slurm
```

**Decode eval (after train):**

```bash
python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-small/100k/tokenized \
  --model-path runs/flan-t5-small/100k-lr5e4/final_model \
  --tokenizer-model google/flan-t5-small \
  --output-json runs/flan-t5-small/100k-lr5e4/test_decode_metrics.json
```

**Prepare after prompt change:**

```bash
python -m main.prepare \
  --output-dir data/processed/flan-t5-base/100k-<tag> \
  --train-size 100k --tokenize \
  --tokenizer-model google/flan-t5-base
```

**Plot runs:**

```bash
python -m main.plot_training_runs --runs-parent runs/flan-t5-base --output runs/flan-t5-base/training_compare.png
```

---

## Out of scope

- Changing ETL / biomarker set without documented ablation
- Training 7B LLMs in this repo track
- Clinician-in-the-loop study (future)
- Alex cluster unless Tier3 access approved

---

*Last updated: 2026-05-23. Plan only — mark `[x]` when done; record numbers in [training_progress.md](training_progress.md).*
