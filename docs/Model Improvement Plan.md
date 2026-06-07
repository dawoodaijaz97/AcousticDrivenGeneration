# Model Improvement Plan

**What this file is:** the **plan** — what to try, in what order, and whether each task is done (`[x]` / `[ ]`).

**What it is not:** numeric results. Those live in **[training_progress.md](training_progress.md)** (summary tables, PD/HC breakdown, conclusions from finished runs).

### Workflow

1. Pick the next open task here (phases, checklist, experiment IDs).
2. Train / `eval_decode` → artifacts under `runs/<model>/<run>/`.
3. **Log metrics** in [training_progress.md](training_progress.md) (new table row + conclusions if useful).
4. **Mark `[x]`** on the matching task and experiment row in this file.

**Related:** [train.md](train.md), [eval-decode.md](eval-decode.md), [data-pipeline.md](data-pipeline.md), [paper-overview.md](paper-overview.md), [hpc-commands.md](hpc-commands.md) (queue, `salloc`, env, `scp`).

**Paper target (LLaMA-7B, ~Table 4 AVG):** ~**0.68**. Current best result: see **B5** in [training_progress.md](training_progress.md).

**Best small @ 100k:** **S4** (`5e-4`, AVG **0.437**) — see [training_progress.md](training_progress.md). **S5** (`3e-4` @ 100k) completed but **below S0**; no further small LR runs planned.

**Best run:** **B11** — flan-t5-base, **`flan-paper`** prompt, LR **5e-4**, **5 epochs**, AVG **0.538** (beats B5 0.529; L5 large @ 5 ep only ties it at 0.536 — size is not the lever; see [training_progress.md](training_progress.md)).

---

## Master checklist

### Pipeline & evaluation discipline

- [x] ETL + canonical splits (`main/` data pipeline)
- [x] Tokenize + train loop (`main.prepare`, `main.train`)
- [x] Decode eval on real test (`main.eval_decode`, beam 3)
- [x] Log PD vs HC in decode JSON (`by_group`)
- [x] Use [training_progress.md](training_progress.md) as the results log (append rows after each eval)
- [x] Compare **checkpoints** vs `final_model` on **B5** (2026-05-29; use `final_model`, not val-loss step)
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
- [ ] Phase 2 prompt variants (category hints, etc.); optional **flan-paper** on **S4** only if time
- [ ] Phase 3 beam sweep on frozen small checkpoint
- [ ] Phase 4 LoRA / `google-t5/t5-small` comparison (if plateau)

### Model size — Flan-T5-base

- [x] **B0** — 100k, LR 3e-4 (AVG **0.395**)
- [x] **B4** — 100k, LR 5e-4 (AVG **0.522**)
- [x] **B5** — Flan paper prefix (`--prompt-style flan-paper`); trained + eval logged (**AVG 0.529**, beats B4)
- [x] Phase 2 — `max_source_length` / `max_target_length` audit on **B5** data (2026-05-29; keep 256/512)
- [x] Phase 3 — beam / checkpoint sweep on **B5** (2026-05-29; keep `final_model` + beam 3)
- [x] Label smoothing ablation (B8/B9/B10) — **closed: confirmed real regression** (2026-05-31; harness verified via B5 sanity decode 0.529, training converged, generation degenerate). **Do not use label smoothing on this recipe.**
- [x] **B11** — epochs 3→5 on B5 recipe (no label smoothing) — **new best AVG 0.538** (beats B5 0.529); underfitting confirmed
- [x] **Large @ 5 epochs (L5)** — done; ties base (0.536 vs 0.538), large not worth its cost
- [x] **B12** — weight-decay **0.0** on B11 recipe — AVG **0.500** (regression vs B11 **0.538**); keep wd **0.01**
- [ ] **B13** — weight-decay **0.05** on B11 recipe — next experiment (complete wd sweep)
- [ ] PD-targeted analysis (PD ~0.50 vs HC ~0.57 across base/large is the bottleneck)

### Model size — Flan-T5-large

- [x] **L0** — 100k, LR 3e-4 (AVG **0.500**) — **reference large config**
- [x] **L4** — 100k, LR 5e-4 (AVG **0.500**, ≈ tie; use L0 for comparisons)
- [x] **L5 — Large @ 5 epochs, flan-paper, 5e-4** (2× A100 / DDP) — **AVG 0.536, ties base B11 (0.538)**; size lever exhausted, `test_loss` 2.78 (overfit/hot LR)
- [ ] Phase 2 prompt experiments (only if base Phase 2 wins) — **deprioritized** (large not worth its cost)

### Cross-cutting (after LR / size baselines)

- [x] Epochs / steps (3 vs 5) — **B11** base 5ep wins (0.538); **L5** large 5ep ties (0.536); size not the lever
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
- [x] Label smoothing trials (B8/B9/B10) — **closed: real regression, do not use** (see [training_progress.md](training_progress.md))
- [x] **Epochs 3→5 (B11)** on B5 recipe — **new best AVG 0.538**
- [x] **Large @ 5 epochs (L5)** — ties base (0.536); size not the lever
- [x] **B12 — weight-decay 0.0 on B11** — AVG **0.500** (regression); keep wd **0.01**
- [ ] **B13 — weight-decay 0.05 on B11** — next experiment
- [ ] Extra weight-decay trials (after B13 if needed)
- [ ] `--no-eval-train` vs default (one 10k run)

### Phase 2 — Prompt and input (re-tokenize required)

- [x] **B5** — Flan paper prefix `Generate a report for:` (`--prompt-style flan-paper`)
- [x] Category hints in prefix (seven mFDA categories; **B6** completed 2026-05-30, below B5)
- [x] Numeric formatting / category labels in feature string (**B7** completed 2026-05-31; below B5)
- [x] `max_source_length` / `max_target_length` — no silent truncation at 256/512 (see [training_progress.md](training_progress.md))
- [x] Re-prepare → re-train **B4** recipe on **flan-paper** prompt — **B5** (`scripts/hpc/train_flan_t5_base_100k_flan_paper_a100.slurm`)

### Phase 3 — Decoding and checkpoints (cheap)

- [x] Beam 1 vs 3 vs 5 on **B5** `final_model` — beam **3** best AVG (**0.5285**)
- [x] Best `checkpoint-*` vs `final_model` — **`final_model`** best; step 74922 ≈ tie; 70000 worse
- [x] Decode settings documented in [training_progress.md](training_progress.md) (beam 3, max 512, ngram 2)

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
| Checkpoint comparison | [x] (B5; `final_model`) |
| Training-time generation metrics | [ ] |

### B. Training hyperparameters

| Experiment | Small | Base | Large |
|------------|-------|------|-------|
| LR sweep | [x] 10k + 100k (S4/S5 logged) | [x] B0 vs B4 | [x] L0 vs L4 |
| Epochs / max-steps | [ ] | [x] B11 (5 ep) = 0.538 | [x] L5 (5 ep) = 0.536 (ties base) |
| Batch + accum | [ ] | [ ] | [ ] |
| Warmup / weight decay | [ ] | [ ] | [ ] |
| Label smoothing | [ ] | [x] **closed: real regression, do not use** (B8/B9/B10) | [ ] |

### C. Prompt and input

| Experiment | Status |
|------------|--------|
| Flan paper prefix (B5) | [x] |
| Category hints in prefix (B6 completed; below B5) | [x] |
| Numeric formatting / category labels (B7 completed; below B5) | [x] |
| `max_source_length` / `max_target_length` audit | [x] (B5 data; 256/512 OK) |

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
| Beam size sweep | [x] (B5: beam 3 best; see training_progress) |
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
| **B5** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper` | **Best overall** AVG **0.529** (+0.007 vs B4) |
| **B6** | flan-t5-base | 100k | flan-paper-categories | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-categories` | Completed; AVG **0.509** (below B5 **0.529**) |
| **B7** | flan-t5-base | 100k | flan-paper-numeric-labels | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-numeric-labels` | Completed; AVG **0.460** (well below B5 **0.529**) |
| **B8** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-ls005` | ls=0.05; **closed** — real regression (AVG ~0.176, degenerate gen), do not use |
| **B9** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-ls010` | ls=0.10; **closed** — real regression (AVG ~0.176, degenerate gen), do not use |
| **B10** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-ls002` | ls=0.02; **closed** — real regression (AVG ~0.176, degenerate gen), do not use |
| **B11** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep` | **New best** AVG **0.538** (5 epochs); beats B5 0.529; `test_loss` 1.094; gain concentrated on HC (0.567 vs 0.538), PD ≈ flat |
| **L5** | flan-t5-large | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-large/100k-flan-paper-5ep` | AVG **0.536** — **ties base B11** (0.538) despite 3× params; HC 0.578 / PD 0.495; `test_loss` 2.78. Size lever exhausted |
| **B12** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-wd0` | wd=**0.0**, 5 ep; AVG **0.500** (vs B11 **0.538**); HC collapse 0.493 vs 0.567; keep wd **0.01** |
| **B13** | flan-t5-base | 100k | flan-paper | 5e-4 | [ ] | `runs/flan-t5-base/100k-flan-paper-5ep-wd005` | wd=**0.05**, 5 ep; complete weight-decay sweep |

---

## Success criteria

| Tier | Small | Base (primary push) |
|------|-------|---------------------|
| **Minimum** | Reproducible decode JSON | Same |
| **Target** | S4: AVG **> 0.362**, HC BLEU **> 0.117** | Phase 2 beat **B4** (0.522) |
| **Stretch** | Narrow gap to paper small-tier | Approach paper ~0.68 band |

---

## Next actions (immediate)

1. [x] **B5** complete — logged in [training_progress.md](training_progress.md).
2. [x] **`max_source_length` / `max_target_length`** audit on B5 data — 0% truncated; keep 256/512.
3. [x] Beam / checkpoint sweep on **B5** — keep **`final_model`**, beam **3** ([training_progress.md](training_progress.md)).
4. [x] Plot: `runs/flan-t5-base/training_compare_b4_b5.png`.
5. [x] **B6** category-hints run complete (re-tokenize + train + decode eval) — did not beat B5.
6. [x] **B7** numeric-formatting variant complete (re-tokenize + train + decode eval) — strong regression vs B5.
7. [x] **B8/B9/B10** label-smoothing runs submitted + decode artifacts logged.
8. [x] B5 sanity decode in same environment → AVG **0.5285** (harness healthy); B8 `trainer_state` converged + decode word-salad → **label smoothing closed as real regression** (see [training_progress.md](training_progress.md)).
9. [x] **B11** (B5 recipe @ 5 epochs) trained + decoded → **new best AVG 0.538** (beats B5 0.529); logged in [training_progress.md](training_progress.md).
10. [x] **L5 — large @ 5 epochs** (flan-paper, 5e-4, 2× A100 DDP) — AVG **0.536**, **ties base B11** (0.538); size not the lever. Decode required `--require-gpu` fix (first attempt CPU-fell-back + hit time limit).
11. [x] **B12 — weight-decay 0.0** on B11 recipe — AVG **0.500** (regression vs B11 **0.538**); keep wd **0.01**. Logged in [training_progress.md](training_progress.md).
12. [ ] **Next:** **B13 — weight-decay 0.05** on B11 recipe (complete wd sweep). If B13 doesn't beat B11, close cheap hyperparameter trials → **Phase 4** (LoRA / freeze-encoder / non-Flan t5) + **PD-targeted analysis**.

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

**L5 prepare (CPU — large flan-paper tokenized data, one-time):**

```bash
cd $WORK/AcousticDrivenGeneration
conda activate acoustic
module load python
export HF_HOME=$WORK/huggingface
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

python -m main.prepare \
  --output-dir data/processed/flan-t5-large/100k-flan-paper \
  --train-size 100k --tokenize \
  --tokenizer-model $WORK/models/flan-t5-large \
  --prompt-style flan-paper
```

**L5 train (2× A100 — DDP via torchrun, 5 epochs):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_large_100k_flan_paper_5ep_2gpu_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_large_100k_flan_paper_5ep_2gpu_a100.slurm
```

**L5 decode eval (GPU — interactive, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-large/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-large/100k-flan-paper-5ep/final_model \
  --tokenizer-model $WORK/models/flan-t5-large \
  --output-json runs/flan-t5-large/100k-flan-paper-5ep/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**B11 train (A100 — epochs 3→5, reuses B5 tokenized data, no label smoothing):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_100k_flan_paper_5ep_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_5ep_a100.slurm
```

**B11 decode eval (GPU — interactive, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**B12 decode eval (GPU — after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-wd0/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-wd0/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**B13 train (A100 — wd 0.05, otherwise B11 recipe):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_100k_flan_paper_5ep_wd005_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_5ep_wd005_a100.slurm
```

**B13 decode eval (GPU — after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-wd005/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-wd005/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**B6 prepare (CPU — category-hint prefix):**

```bash
cd $WORK/AcousticDrivenGeneration
conda activate acoustic
module load python
export HF_HOME=$WORK/huggingface
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

python -m main.prepare \
  --output-dir data/processed/flan-t5-base/100k-flan-paper-categories \
  --train-size 100k --tokenize \
  --tokenizer-model $WORK/models/flan-t5-base \
  --prompt-style flan-paper-categories
```

**B6 train (A100 — Slurm only):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_100k_flan_paper_categories_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_categories_a100.slurm
```

**B6 decode eval (GPU — interactive, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper-categories/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-categories/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-categories/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**B7 prepare (CPU — numeric labels + deterministic formatting):**

```bash
cd $WORK/AcousticDrivenGeneration
conda activate acoustic
module load python
export HF_HOME=$WORK/huggingface
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

python -m main.prepare \
  --output-dir data/processed/flan-t5-base/100k-flan-paper-numeric-labels \
  --train-size 100k --tokenize \
  --tokenizer-model $WORK/models/flan-t5-base \
  --prompt-style flan-paper-numeric-labels
```

**B7 train (A100 — Slurm only):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_100k_flan_paper_numeric_labels_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_numeric_labels_a100.slurm
```

**B7 decode eval (GPU — interactive, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper-numeric-labels/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-numeric-labels/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-numeric-labels/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**B8 train (A100 — label smoothing 0.05 on B5 tokenized data):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_100k_flan_paper_ls005_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_ls005_a100.slurm
```

**B8 decode eval (GPU — interactive, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-ls005/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-ls005/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**B9 train (A100 — label smoothing 0.10 on B5 tokenized data):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_100k_flan_paper_ls010_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_ls010_a100.slurm
```

**B9 decode eval (GPU — interactive, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-ls010/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-ls010/test_decode_metrics.json \
  --batch-size 8 --seed 42
```

**B10 train (A100 — label smoothing 0.02 on B5 tokenized data):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_100k_flan_paper_ls002_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_ls002_a100.slurm
```

**B10 decode eval (GPU — interactive, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-ls002/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-ls002/test_decode_metrics.json \
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

*Last updated: 2026-06-06. B11 (wd 0.01, 5 ep) remains best (AVG 0.538). B12 (wd 0.0) regressed to 0.500. Next: B13 (wd 0.05), then Phase 4 / PD analysis. Plan only — mark `[x]` when done; record numbers in [training_progress.md](training_progress.md).*
