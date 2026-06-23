# Model Improvement Plan

**What this file is:** the **plan** — what to try, in what order, and whether each task is done (`[x]` / `[ ]`).

**What it is not:** numeric results. Those live in **[training_progress.md](training_progress.md)** (summary tables, PD/HC breakdown, conclusions from finished runs).

### Workflow

1. Pick the next open task here (phases, checklist, experiment IDs).
2. Train / `eval_decode` → artifacts under `runs/<model>/<run>/`.
3. **Log metrics** in [training_progress.md](training_progress.md) (new table row + conclusions if useful).
4. **Mark `[x]`** on the matching task and experiment row in this file.

**Related:** [train.md](train.md), [eval-decode.md](eval-decode.md), [data-pipeline.md](data-pipeline.md), [paper-overview.md](paper-overview.md), [hpc-commands.md](hpc-commands.md) (queue, `salloc`, env, `scp`).

**Paper target (LLaMA-7B, ~Table 4 AVG):** ~**0.68**. Current best result: **B17** (AVG **0.542**; **B18** ≈ tie) in [training_progress.md](training_progress.md).

**Best small @ 100k:** **S4** (`5e-4`, AVG **0.437**) — see [training_progress.md](training_progress.md). **S5** (`3e-4` @ 100k) completed but **below S0**; no further small LR runs planned.

**Best run:** **B17** — B11 checkpoint + **LoRA rank 32**, 3 ep, LR **5e-4**, AVG **0.542** (beats B14 **0.540** and B11 **0.538**; see [training_progress.md](training_progress.md)). **B18** (LoRA r=32 on B17, 5 ep) ≈ **tie** (AVG **0.542**); **longer LoRA lever closed**. **Best full fine-tune:** **B11** (0.538).

### 2026-06-24 diagnosis — plateau is structural, not just training

The current plateau is not primarily a model-size or learning-rate problem. Base, large, LoRA rank, longer LoRA, freeze-encoder, weight decay, prompt-only template, label smoothing, and severity oversampling have all been tested and are either closed or tied. The next bottleneck is **schema reliability**: decoded reports need to consistently preserve all seven mFDA categories.

**Primary failure mode:** generated reports are report-like, but they do not reliably emit the full seven-slot clinical structure. The PD structure analysis showed category coverage around **0.143**, with **Breathing** dominating and the remaining categories often missing. Therefore, the next plan should prioritize **constrained decoding, structure-aware reranking, and structured targets** before spending more GPU time on normal fine-tuning.

**Immediate priority order:**

1. ~~**B21:** B17 + forced seven-slot post-processing~~ — **done; closed** (coverage **1.0**, AVG **0.492** vs B17 **0.542**; see [training_progress.md](training_progress.md)).
2. ~~**B22:** B17 + multi-candidate generation + structure-aware reranking~~ — **done; closed** (coverage **0.144** unchanged, AVG **0.531** vs B17 **0.542**; see [training_progress.md](training_progress.md)).
3. **D1:** synthetic-vs-real target/report distribution audit.
4. **B23:** train a structured seven-label target format, then verbalize through a deterministic report template.

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
- [x] Per-category keyword/slot checks on decoded text — **B14 PD analysis** (`main/analyze_pd_decode`, `pd_analysis.json`)
- [x] Add schema/slot coverage to standard eval outputs (`category_coverage`, `severity_word_coverage`, `all_7_slots_rate`) — in `main.structured_decode` JSON (B21 logged)
- [x] Add structure-aware reranking utility for multi-candidate decode outputs (`main.structured_decode --structure-rerank`; B22 eval **closed**)
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
- [ ] Phase 4 LoRA on **Flan-T5** (if plateau); non-Flan **`google-t5/t5-small`** deprioritized after **N0**

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
- [x] **B13** — weight-decay **0.05** on B11 recipe — AVG **0.517** (regression vs B11 **0.538**); **weight-decay sweep closed**, keep wd **0.01**
- [x] **N0** — non-Flan **`google-t5/t5-base`**, B11 recipe — AVG **0.439** (regression vs B11 **0.538**); **non-Flan base closed**
- [x] **B14** — LoRA rank **16** on B11 `final_model`, 3 ep — AVG **0.540** (PD **0.512**, HC **0.568**)
- [x] **B16** — LoRA rank **8** on B11 `final_model`, 3 ep — AVG **0.538** (ties B11; below B14)
- [x] **B17** — LoRA rank **32** on B11 `final_model`, 3 ep — AVG **0.542** (**new best overall**; PD **0.513**, HC **0.571**)
- [x] **B18** — LoRA rank **32** on B17 `final_model`, **5 ep** — AVG **0.542** (≈ tie B17 **0.542**; PD **0.514**, HC **0.570**); **longer LoRA lever closed**
- [x] **B15** — freeze encoder on B11 `final_model`, 3 ep — AVG **0.529** (below B11 **0.538**); **freeze-encoder lever closed**
- [x] PD-targeted analysis on **B14** — structural slot check (`pd_analysis.json`); see [training_progress.md](training_progress.md)
- [x] **B19** — `flan-paper-report-template` prompt + LoRA r=32 on B17, 3 ep — AVG **0.540** (below B17 **0.542**); category coverage **≈14%** unchanged; **template prompt closed**
- [x] **B20** — **Moderate+ severity 2× oversample** + LoRA r=32 on B17, 3 ep — AVG **0.542** (≈ tie B17); PD **0.513**; coverage **0.143** unchanged; **oversampling closed**
- [x] **B21** — B17 + **forced seven-slot constrained/post-processed output**; no retraining; compare raw vs structured decode on real test — **closed:** coverage **1.0**, structured AVG **0.492** (raw **0.542** ≈ B17)
- [x] **B22** — B17 + **multi-candidate generation + structure-aware reranking**; no retraining — **closed:** coverage **0.144** unchanged, AVG **0.531** (below B17 **0.542**)
- [ ] **B23** — train **structured seven-label target** (`Breathing: Mild`, ..., `Intelligibility: Normal`) and generate final prose with deterministic template

### Model size — Flan-T5-large

- [x] **L0** — 100k, LR 3e-4 (AVG **0.500**) — **reference large config**
- [x] **L4** — 100k, LR 5e-4 (AVG **0.500**, ≈ tie; use L0 for comparisons)
- [x] **L5 — Large @ 5 epochs, flan-paper, 5e-4** (2× A100 / DDP) — **AVG 0.536, ties base B11 (0.538)**; size lever exhausted, `test_loss` 2.78 (overfit/hot LR)
- [ ] Phase 2 prompt experiments (only if base Phase 2 wins) — **deprioritized** (large not worth its cost)

### Cross-cutting (after LR / size baselines)

- [x] Epochs / steps (3 vs 5) — **B11** base 5ep wins (0.538); **L5** large 5ep ties (0.536); size not the lever
- [ ] Batch + gradient accumulation sweep
- [x] Warmup / weight decay trials — **closed:** B12 (wd=0.0, 0.500) and B13 (wd=0.05, 0.517) both below B11 (wd=0.01, **0.538**); label smoothing closed separately (B8/B9/B10)
- [ ] `--no-eval-train` wall-clock trial (one 10k run)
- [ ] FP16 on A100 only (V100 had NaN with fp16)
- [ ] Leakage audit (`example_hash` train/val/test)
- [ ] **D1 synthetic-vs-real audit:** compare target/report structure, category frequency, severity distribution, phrase repetition, and PD/HC wording drift
- [ ] Use structure metrics as promotion gates: do not promote a run that improves AVG while keeping category coverage near **0.143**

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
- [x] **B13 — weight-decay 0.05 on B11** — AVG **0.517** (regression); **wd sweep closed**
- [x] Extra weight-decay trials — **closed** (0.01 best; 0.0 and 0.05 regress)
- [ ] `--no-eval-train` vs default (one 10k run)

### Phase 2 — Prompt and input (re-tokenize required)

- [x] **B5** — Flan paper prefix `Generate a report for:` (`--prompt-style flan-paper`)
- [x] Category hints in prefix (seven mFDA categories; **B6** completed 2026-05-30, below B5)
- [x] Numeric formatting / category labels in feature string (**B7** completed 2026-05-31; below B5)
- [x] **B19** — seven-slot `Category (Severity):` output template in prefix (`flan-paper-report-template`) — AVG **0.540**, coverage **≈14%** unchanged vs B17; **closed**
- [x] `max_source_length` / `max_target_length` — no silent truncation at 256/512 (see [training_progress.md](training_progress.md))
- [x] Re-prepare → re-train **B4** recipe on **flan-paper** prompt — **B5** (`scripts/hpc/train_flan_t5_base_100k_flan_paper_a100.slurm`)

### Phase 3 — Decoding and checkpoints (cheap)

- [x] Beam 1 vs 3 vs 5 on **B5** `final_model` — beam **3** best AVG (**0.5285**)
- [x] Best `checkpoint-*` vs `final_model` — **`final_model`** best; step 74922 ≈ tie; 70000 worse
- [x] Decode settings documented in [training_progress.md](training_progress.md) (beam 3, max 512, ngram 2)

### Phase 4 — Efficiency (if quality plateaus)

- [x] LoRA / adapters (`peft`, rank 8–32) on **Flan-T5-base (B11)** — **B17** r=32 **AVG 0.542** best; rank sweep **closed**
- [x] Freeze encoder, train decoder on **Flan-T5-base (B11)** — **B15** AVG **0.529**, below B11; **closed**
- [x] Optional LoRA rank sweep (8 / 16 / 32) on B11 checkpoint — **B16 0.538**, **B14 0.540**, **B17 0.542**; **closed**
- [x] **B18** — longer LoRA (r=32, **5 ep**) on B17 `final_model` — AVG **0.542** ≈ B17; **closed**
- [x] Compare **`google-t5/t5-base`** (non-Flan) at same pipeline — **N0** AVG **0.439**, well below B11 **0.538**; **closed**
- [ ] Optional: **`google-t5/t5-small`** (non-Flan) at same pipeline — deprioritized after N0

### Phase 5 — Structure-forcing and schema reliability (current priority)

**Goal:** move from free-form report-like text to clinically valid seven-category mFDA output. A run should now be judged on both standard generation metrics and structure metrics.

- [x] **B21 — constrained seven-slot output, no retraining** — **closed (2026-06-24)**
  - Source model: `runs/flan-t5-base/100k-flan-paper-5ep-lora32/final_model` (**B17**)
  - Method: decode normally, then normalize into a mandatory seven-category skeleton
  - **Result:** `category_coverage` **0.144 → 1.0**, `all_7_slots_rate` **0 → 1.0**, but structured AVG **0.492** vs raw **0.542** (−**0.050**) — **promotion gate failed**; padded `Normal: Within normal limits.` slots hurt ROUGE/BLEU vs full reference prose
  - Artifacts: `runs/flan-t5-base/100k-flan-paper-5ep-lora32-structured-decode/`

- [x] **B22 — structure-aware reranking, no retraining** — **closed (2026-06-24)**
  - Source model: **B17**; beams **8**, **5** return sequences per item
  - **Result:** **`category_coverage` 0.144** / **`all_7_slots_rate` 0.0** (≈ B17); AVG **0.531** vs B17 **0.542** (−**0.011**) — candidates never contain fuller 7-slot structure; rerank cannot help
  - Artifacts: `runs/flan-t5-base/100k-flan-paper-5ep-lora32-rerank/`

- [ ] **D1 — synthetic-vs-real report distribution audit**
  - Compare train/val synthetic targets vs real test references
  - Check category heading frequency, severity distribution per category, report length, repeated phrases, PD/HC wording drift, and whether synthetic targets overrepresent Breathing-like wording
  - Output artifact: `analysis/synthetic_real_report_audit.json` plus a short Markdown summary

- [ ] **B23 — structured target training**
  - Replace prose target with seven-label target format:

    ```text
    Breathing: Mild
    Lips: Normal
    Larynx: Moderate
    Palate: Normal
    Monotonicity: Mild
    Tongue: Moderate
    Intelligibility: Mild
    ```

  - Train/evaluate as structured prediction first; then verbalize using deterministic clinical report templates
  - Track both label accuracy/F1 per category and final text metrics after verbalization

**Do not resume broad training sweeps until D1/B23 clarify the structural failure mode** (B21 post-process and B22 rerank **closed** at decode time).

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
| Category coverage / slot coverage metrics | [ ] |
| Structure-aware promotion gate | [ ] |

### B. Training hyperparameters

| Experiment | Small | Base | Large |
|------------|-------|------|-------|
| LR sweep | [x] 10k + 100k (S4/S5 logged) | [x] B0 vs B4 | [x] L0 vs L4 |
| Epochs / max-steps | [ ] | [x] B11 (5 ep) = 0.538 | [x] L5 (5 ep) = 0.536 (ties base) |
| Batch + accum | [ ] | [ ] | [ ] |
| Warmup / weight decay | [ ] | [x] **closed:** B11 wd=0.01 best (0.538); B12 wd=0.0 (0.500); B13 wd=0.05 (0.517) | [ ] |
| Label smoothing | [ ] | [x] **closed: real regression, do not use** (B8/B9/B10) | [ ] |

### C. Prompt and input

| Experiment | Status |
|------------|--------|
| Flan paper prefix (B5) | [x] |
| Category hints in prefix (B6 completed; below B5) | [x] |
| Numeric formatting / category labels (B7 completed; below B5) | [x] |
| Seven-slot output template in prefix (B19) | [x] (below B17; coverage unchanged) |
| Forced seven-slot output after decode (B21) | [x] (closed: coverage 1.0, AVG 0.492) |
| Multi-candidate structure-aware reranking (B22) | [x] (closed: coverage 0.144, AVG 0.531) |
| Structured seven-label target format (B23) | [ ] |
| `max_source_length` / `max_target_length` audit | [x] (B5 data; 256/512 OK) |

### D. Data and splits

| Experiment | Status |
|------------|--------|
| Train size 10k vs 100k | [x] (both available) |
| Fast 1k smoke runs | [x] |
| `google-t5/t5-base` comparison | [x] **N0** AVG **0.439** vs B11 **0.538** — closed |
| `google-t5/t5-small` comparison | [ ] deprioritized after N0 |
| PD group oversampling (B20) | [x] (severity proxy; ≈ tie B17; coverage unchanged) |
| `--no-eval-train` trial | [ ] |
| Leakage audit | [ ] |
| Synthetic-vs-real target/report distribution audit (D1) | [ ] |

### E. Model efficiency

| Action | Status |
|--------|--------|
| LoRA / adapters | [x] **B14** r=16 → 0.540; **B16** r=8 → 0.538; **B17** r=32 → **0.542** (best); sweep closed |
| Freeze encoder | [x] **B15** on B11 → AVG **0.529** (closed) |
| Span-corruption pretrain | [ ] (out of scope unless plateau) |
| FP16 on A100 | [ ] |

### F. Decoding knobs

| Knob | Status |
|------|--------|
| Beam size sweep | [x] (B5: beam 3 best; see training_progress) |
| `max_new_tokens` ~512 | [x] |
| `no_repeat_ngram` 2 | [x] (paper-aligned in eval) |
| Constrained seven-slot template output | [x] B21 (closed) |
| Structure-aware candidate reranking | [x] B22 (closed) |

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
| **B13** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-wd005` | wd=**0.05**, 5 ep; AVG **0.517** (vs B11 **0.538**); PD 0.486 / HC 0.548; wd sweep closed |
| **N0** | t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/t5-base/100k-flan-paper-5ep` | non-Flan **`google-t5/t5-base`**, B11 recipe; AVG **0.439** (−0.099 vs B11); HC 0.429 / PD 0.448; **non-Flan base closed** |
| **B14** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-lora16` | LoRA **r=16** on B11 `final_model`, 3 ep; AVG **0.540**; PD **0.512** / HC **0.568** |
| **B16** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-lora8` | LoRA **r=8** on B11 `final_model`, 3 ep; AVG **0.538** (≈ B11); PD **0.509** / HC **0.567** |
| **B17** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-lora32` | LoRA **r=32** on B11 `final_model`, 3 ep; AVG **0.542** (**new best**); PD **0.513** / HC **0.571**; LoRA rank sweep closed |
| **B18** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-lora32-5ep` | LoRA **r=32** on B17 `final_model`, **5 ep**; AVG **0.542** (≈ tie B17); PD **0.514** / HC **0.570**; **longer LoRA closed** |
| **B19** | flan-t5-base | 100k | flan-paper-report-template | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-report-template-lora32` | LoRA **r=32** on B17, 3 ep; AVG **0.540** (vs B17 **0.542**); PD **0.509** / HC **0.572**; `pd_analysis` coverage **0.143** (≈ B17); **template prompt closed** |
| **B20** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x` | LoRA **r=32** on B17, 3 ep; **`--oversample-severity-min Moderate --oversample-factor 2`**; AVG **0.542** (≈ B17); PD **0.513**; coverage **0.143** unchanged; **closed** |
| **B15** | flan-t5-base | 100k | flan-paper | 5e-4 | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-freeze-enc` | **`--freeze-encoder`** on B11 `final_model`, 3 ep; AVG **0.529** (vs B11 **0.538**); PD **0.499** / HC **0.560**; **freeze-encoder closed** |
| **B21** | flan-t5-base | 100k | flan-paper | n/a | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-lora32-structured-decode` | **No retraining.** B17 + forced seven-slot post-process — coverage **1.0**, structured AVG **0.492** (raw **0.542**); **closed** |
| **B22** | flan-t5-base | 100k | flan-paper | n/a | [x] | `runs/flan-t5-base/100k-flan-paper-5ep-lora32-rerank` | **No retraining.** B17 + structure rerank — coverage **0.144** unchanged, AVG **0.531**; **closed** |
| **D1** | data audit | train/val/test | n/a | n/a | [ ] | `analysis/synthetic_real_report_audit.json` | Synthetic-vs-real report/target distribution audit before more training |
| **B23** | flan-t5-base | 100k | structured-seven-label | 5e-4 | [ ] | `runs/flan-t5-base/100k-structured-seven-label` | Train structured category/severity labels first; deterministic template verbalization after |

---

## Success criteria

| Tier | Small | Base (primary push) |
|------|-------|---------------------|
| **Minimum** | Reproducible decode JSON | Same |
| **Target** | S4: AVG **> 0.362**, HC BLEU **> 0.117** | Beat B17 **0.542** while improving schema coverage |
| **Structure target** | n/a | Category coverage **0.143 → ≥0.90**, `all_7_slots_rate` clearly above B17 |
| **Stretch** | Narrow gap to paper small-tier | Approach paper ~0.68 band **with valid seven-category reports** |

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
12. [x] **B13 — weight-decay 0.05** on B11 recipe — AVG **0.517** (regression vs B11 **0.538**); **weight-decay sweep closed**. Logged in [training_progress.md](training_progress.md).
13. [x] **N0 — non-Flan `google-t5/t5-base`**, B11 recipe — AVG **0.439** (regression vs B11 **0.538**); **non-Flan base closed**. Logged in [training_progress.md](training_progress.md).
14. [x] **B14 — LoRA rank 16 on B11 `final_model`**, 3 ep — AVG **0.540** (**new best overall**; +0.002 vs B11); PD **0.512**, HC **0.568**. Logged in [training_progress.md](training_progress.md).
15. [x] **B15 — freeze encoder on B11 `final_model`**, 3 ep — AVG **0.529** (below B11 **0.538**); **freeze-encoder lever closed**. Logged in [training_progress.md](training_progress.md).
16. [x] **PD-targeted analysis on B14** — `pd_analysis.json`; models rarely emit full 7-slot `Category (Severity):` template (~14% parsed coverage); Breathing dominates; B14 vs B11: PD severity match slightly worse, HC slightly better. Logged in [training_progress.md](training_progress.md).
17. [x] **B16 — LoRA rank 8** on B11 — AVG **0.538** (≈ tie B11); **B17 — LoRA rank 32** — AVG **0.542** (**new best**). **LoRA rank sweep closed.**
18. [x] **B18 — LoRA rank 32 on B17 `final_model`, 5 ep** — AVG **0.542** (≈ tie B17 **0.542**); PD **0.514** / HC **0.570**. **Longer LoRA lever closed.** Logged in [training_progress.md](training_progress.md).
19. [x] **B19 — `flan-paper-report-template` + LoRA r=32 on B17**, 3 ep — AVG **0.540** (below B17 **0.542**); `pd_analysis` category coverage **0.143** (≈ unchanged); **template prompt lever closed.** Logged in [training_progress.md](training_progress.md).
20. [x] **B20 — Moderate+ severity 2× oversample + LoRA r=32 on B17**, 3 ep — AVG **0.542** (≈ tie B17); `pd_analysis` coverage **0.143** unchanged. **Severity oversampling closed.** Logged in [training_progress.md](training_progress.md).
21. [x] **B21 — constrained seven-slot output** using B17 predictions; no retraining. **Closed:** coverage **1.0**, AVG **0.492** (raw **0.542**). Logged in [training_progress.md](training_progress.md).
22. [x] **B22 — structure-aware reranking** using B17 multi-candidate generation. **Closed:** coverage **0.144**, AVG **0.531** (below B17 **0.542**). Logged in [training_progress.md](training_progress.md).
23. [ ] **D1 — synthetic-vs-real report audit** before more training; confirm whether synthetic targets teach the same category/wording distribution as real reports.
24. [ ] **B23 — structured seven-label target training** — next after **D1** (B21/B22 decode levers **closed**).
25. [x] Add `category_coverage`, `severity_word_coverage`, and `all_7_slots_rate` to the standard logging workflow — via `main.structured_decode` (B21).

---

## Commands cheat sheet

**B21/B22 implementation note:** `main.structured_decode` — B21 **closed** (forced template); B22 **closed** (structure rerank).

**B21 decode eval target (GPU — no retraining, after utility is implemented):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.structured_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora32/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-structured-decode/test_decode_metrics.json \
  --output-predictions-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-structured-decode/test_decode_predictions.json \
  --force-seven-slot-template \
  --batch-size 8 --seed 42 \
  --require-gpu
```

**B22 structure-aware reranking target (GPU — no retraining, after utility is implemented):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.structured_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora32/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-rerank/test_decode_metrics.json \
  --output-predictions-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-rerank/test_decode_predictions.json \
  --num-beams 8 \
  --num-return-sequences 5 \
  --structure-rerank \
  --batch-size 4 --seed 42 \
  --require-gpu
```

**D1 synthetic-vs-real report audit target:**

```bash
python -m main.audit_report_distribution \
  --train data/Data_splits/100k_samples_balanced/train_split.csv \
  --val data/Data_splits/val_split.csv \
  --test data/Data_splits/test_split.csv \
  --output-json analysis/synthetic_real_report_audit.json \
  --output-md analysis/synthetic_real_report_audit.md
```

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

**N0 prepare (CPU — non-Flan t5-base; download `google-t5/t5-base` to `$WORK/models/t5-base` first):**

```bash
cd $WORK/AcousticDrivenGeneration
conda activate acoustic
module load python
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME=$WORK/huggingface
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

python -m main.prepare \
  --output-dir data/processed/t5-base/100k-flan-paper \
  --train-size 100k --tokenize \
  --tokenizer-model $WORK/models/t5-base \
  --prompt-style flan-paper
```

**N0 train (A100 — Slurm):**

```bash
sed -i 's/\r$//' scripts/hpc/train_t5_base_100k_flan_paper_5ep_a100.slurm
sbatch.tinygpu scripts/hpc/train_t5_base_100k_flan_paper_5ep_a100.slurm
```

**N0 decode eval (GPU — interactive on compute node, after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/t5-base/100k-flan-paper/tokenized \
  --model-path runs/t5-base/100k-flan-paper-5ep/final_model \
  --tokenizer-model $WORK/models/t5-base \
  --output-json runs/t5-base/100k-flan-paper-5ep/test_decode_metrics.json \
  --batch-size 8 --seed 42 \
  --require-gpu
```

**B14 train (A100 — LoRA r=16 on B11 `final_model`, 3 ep):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_b14_lora16_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_b14_lora16_a100.slurm
```

**B14 decode eval (GPU — after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora16/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora16/test_decode_metrics.json \
  --batch-size 8 --seed 42 \
  --require-gpu
```

**B18 train (A100 — LoRA r=32 on B17 `final_model`, 5 ep):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_b18_lora32_5ep_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_b18_lora32_5ep_a100.slurm
```

**B18 decode eval (GPU — after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora32-5ep/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-5ep/test_decode_metrics.json \
  --batch-size 8 --seed 42 \
  --require-gpu
```

Or batch eval via Slurm:

```bash
sed -i 's/\r$//' scripts/hpc/eval_flan_t5_base_b18_lora32_5ep_a100.slurm
sbatch.tinygpu scripts/hpc/eval_flan_t5_base_b18_lora32_5ep_a100.slurm
```

**B19 prepare (CPU — `flan-paper-report-template`, re-tokenize required):**

```bash
cd $WORK/AcousticDrivenGeneration
conda activate acoustic
module load python
export HF_HOME=$WORK/huggingface
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

sed -i 's/\r$//' scripts/hpc/prepare_flan_t5_base_b19_report_template.sh
bash scripts/hpc/prepare_flan_t5_base_b19_report_template.sh
```

**B19 train (A100 — LoRA r=32 on B17 `final_model`, 3 ep, new tokenized data):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_b19_report_template_lora32_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_b19_report_template_lora32_a100.slurm
```

**B19 decode eval + PD structure analysis (GPU — after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper-report-template/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-report-template-lora32/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-report-template-lora32/test_decode_metrics.json \
  --output-predictions-json runs/flan-t5-base/100k-flan-paper-report-template-lora32/test_decode_predictions.json \
  --batch-size 8 --seed 42 \
  --require-gpu

python -m main.analyze_pd_decode \
  --predictions-json runs/flan-t5-base/100k-flan-paper-report-template-lora32/test_decode_predictions.json \
  --output-json runs/flan-t5-base/100k-flan-paper-report-template-lora32/pd_analysis.json
```

Or batch eval via Slurm:

```bash
sed -i 's/\r$//' scripts/hpc/eval_flan_t5_base_b19_report_template_lora32_a100.slurm
sbatch.tinygpu scripts/hpc/eval_flan_t5_base_b19_report_template_lora32_a100.slurm
```

**B20 train (A100 — PD 2× oversample + LoRA r=32 on B17, 3 ep; reuses B5 tokenized data):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_b20_pd_oversample_lora32_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_b20_pd_oversample_lora32_a100.slurm
```

**B20 decode eval + PD structure analysis (GPU — after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/test_decode_metrics.json \
  --output-predictions-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/test_decode_predictions.json \
  --batch-size 8 --seed 42 \
  --require-gpu

python -m main.analyze_pd_decode \
  --predictions-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/test_decode_predictions.json \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32-pd-2x/pd_analysis.json
```

Or batch eval via Slurm:

```bash
sed -i 's/\r$//' scripts/hpc/eval_flan_t5_base_b20_pd_oversample_lora32_a100.slurm
sbatch.tinygpu scripts/hpc/eval_flan_t5_base_b20_pd_oversample_lora32_a100.slurm
```

**B15 train (A100 — freeze encoder on B11 `final_model`, 3 ep):**

```bash
sed -i 's/\r$//' scripts/hpc/train_flan_t5_base_b15_freeze_enc_a100.slurm
sbatch.tinygpu scripts/hpc/train_flan_t5_base_b15_freeze_enc_a100.slurm
```

**B15 decode eval (GPU — after train):**

```bash
export HF_HOME=$WORK/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m main.eval_decode \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --model-path runs/flan-t5-base/100k-flan-paper-5ep-freeze-enc/final_model \
  --tokenizer-model $WORK/models/flan-t5-base \
  --output-json runs/flan-t5-base/100k-flan-paper-5ep-freeze-enc/test_decode_metrics.json \
  --batch-size 8 --seed 42 \
  --require-gpu
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
- More broad LR/epoch/model-size sweeps before **D1/B23** address the structural failure mode (B21/B22 **closed**)
- Training 7B LLMs in this repo track
- Clinician-in-the-loop study (future)
- Alex cluster unless Tier3 access approved

---

*Last updated: 2026-06-24. **B17** remains best reporting config (AVG **0.542**). **B21/B22 closed** at decode time (structure not fixable without retraining). Next: **D1 → B23**. Plan only — mark `[x]` when done; record numbers in [training_progress.md](training_progress.md).*
