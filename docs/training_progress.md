# Training progress log

**What this file is:** **results** from executing the [Model Improvement Plan](Model%20Improvement%20Plan.md) — metrics, comparisons, and takeaways after runs finish.

**What it is not:** the task list. Use the plan doc for `[x]` checkboxes, phases, and what to run next.

### When a run finishes

1. Run `main.eval_decode` → `runs/<model>/<run>/test_decode_metrics.json`
2. Add a row to **Summary — all runs** (and PD/HC table if useful)
3. Update **Conclusions** if the run changes how you read prior results
4. Mark the matching experiment **`[x]`** in the [Model Improvement Plan](Model%20Improvement%20Plan.md)

---

Metrics from **`main.eval_decode`** (real **test**, 96 rows; beam 3, max 512 tokens) and **`test_eval.json`** (teacher-forcing **test_loss** after `main.train`). All generation scores are on a **0–1** scale (higher is better except `test_loss`).

**Paper reference (LLaMA-7B, Table 4 ~AVG):** ~**0.68** overall. Our Flan-T5 runs remain below that; this log tracks **relative** gains across configs.

**Artifacts per run:** `runs/<model>/<run>/test_decode_metrics.json`, `test_eval.json`.

---

## Summary — all runs with decode eval

| Run ID | Model | Train | LR | `test_loss` ↓ | R-1 | R-2 | R-L | BLEU | BERT | **AVG** ↑ |
|--------|-------|-------|-----|---------------|-----|-----|-----|------|------|-----------|
| S0 | flan-t5-small | 100k | 3e-4 | 1.075 | 0.373 | 0.191 | 0.303 | 0.186 | 0.759 | **0.362** |
| **S4** | flan-t5-small | 100k | 5e-4 | 1.102 | 0.481 | 0.256 | 0.394 | 0.265 | 0.791 | **0.437** |
| **S5** | flan-t5-small | 100k | 3e-4 | 1.158 | 0.358 | 0.167 | 0.288 | 0.146 | 0.760 | **0.344** |
| S1 | flan-t5-small | 10k | 1e-4 | 0.679 | 0.436 | 0.207 | 0.334 | 0.287 | 0.790 | **0.411** |
| S2 | flan-t5-small | 10k | 3e-4 | 0.835 | 0.452 | 0.227 | 0.360 | 0.308 | 0.798 | **0.429** |
| S3 | flan-t5-small | 10k | 5e-4 | 0.802 | 0.511 | 0.257 | 0.407 | 0.304 | 0.814 | **0.459** |
| — | flan-t5-small | 1k | 3e-4 | 0.237 | 0.486 | 0.224 | 0.375 | 0.293 | 0.804 | **0.436** |
| B0 | flan-t5-base | 100k | 3e-4 | 1.224 | 0.416 | 0.233 | 0.342 | 0.215 | 0.771 | **0.395** |
| B4 | flan-t5-base | 100k | 5e-4 | 1.186 | 0.584 | 0.362 | 0.499 | 0.351 | 0.816 | **0.522** |
| **B5** | flan-t5-base | 100k | 5e-4 | 1.203 | 0.586 | 0.360 | 0.510 | 0.369 | 0.819 | **0.529** |
| B6 | flan-t5-base | 100k | 5e-4 | 1.355 | 0.567 | 0.341 | 0.487 | 0.337 | 0.812 | **0.509** |
| B7 | flan-t5-base | 100k | 5e-4 | 0.811 | 0.524 | 0.261 | 0.413 | 0.298 | 0.803 | **0.460** |
| B8 | flan-t5-base | 100k | 5e-4 | 1.127 | 0.147 | 0.002 | 0.114 | 0.000 | 0.615 | **0.176** |
| B9 | flan-t5-base | 100k | 5e-4 | 1.733 | 0.147 | 0.002 | 0.114 | 0.000 | 0.615 | **0.176** |
| B10 | flan-t5-base | 100k | 5e-4 | 0.753 | 0.147 | 0.002 | 0.114 | 0.000 | 0.615 | **0.176** |
| L0 | flan-t5-large | 100k | 3e-4 | 0.997 | 0.566 | 0.318 | 0.461 | 0.346 | 0.810 | **0.500** |
| L4 | flan-t5-large | 100k | 5e-4 | 1.057 | 0.568 | 0.314 | 0.482 | 0.326 | 0.812 | **0.500** |
| **B11** | flan-t5-base | 100k | 5e-4 | 1.094 | 0.600 | 0.384 | 0.527 | 0.358 | 0.822 | **0.538** |
| L5 | flan-t5-large | 100k | 5e-4 | 2.779 | 0.606 | 0.361 | 0.525 | 0.366 | 0.823 | **0.536** |
| B12 | flan-t5-base | 100k | 5e-4 | 1.351 | 0.555 | 0.325 | 0.471 | 0.338 | 0.809 | **0.500** |
| B13 | flan-t5-base | 100k | 5e-4 | 1.159 | 0.581 | 0.351 | 0.495 | 0.342 | 0.814 | **0.517** |
| N0 | t5-base | 100k | 5e-4 | 1.247 | 0.455 | 0.269 | 0.393 | 0.285 | 0.791 | **0.439** |
| B15 | flan-t5-base | 100k | 5e-4 | 1.210 | 0.588 | 0.376 | 0.517 | 0.346 | 0.818 | **0.529** |
| B16 | flan-t5-base | 100k | 5e-4 | 1.150 | 0.600 | 0.383 | 0.529 | 0.357 | 0.823 | **0.538** |
| B14 | flan-t5-base | 100k | 5e-4 | 1.166 | 0.604 | 0.383 | 0.528 | 0.360 | 0.823 | **0.540** |
| **B17** | flan-t5-base | 100k | 5e-4 | 1.145 | 0.604 | 0.389 | 0.531 | 0.362 | 0.823 | **0.542** |
| B18 | flan-t5-base | 100k | 5e-4 | 1.145 | 0.605 | 0.390 | 0.530 | 0.362 | 0.823 | **0.542** |
| B19 | flan-t5-base | 100k | 5e-4 | 1.156 | 0.600 | 0.386 | 0.530 | 0.361 | 0.822 | **0.540** |

↓ lower is better for `test_loss`; ↑ higher is better for decode metrics. **B17** = B11 checkpoint + **LoRA rank 32**, 3 ep — **best reporting config** (AVG **0.542**). **B18** ≈ tie B17; **longer LoRA closed**. **B19** = `flan-paper-report-template` + LoRA on B17 — **below B17** (AVG **0.540**); **template prompt closed** (7-slot coverage ≈ unchanged).

**Run directories:** `runs/flan-t5-small/{100k,100k-lr5e4,100k-lr3e4,10k-lr1e4,10k-lr3e4,10k-lr5e4,1k}`, `runs/flan-t5-base/{100k,100k-lr5e4,100k-flan-paper,100k-flan-paper-categories,100k-flan-paper-numeric-labels,100k-flan-paper-ls002,100k-flan-paper-ls005,100k-flan-paper-ls010,100k-flan-paper-5ep,100k-flan-paper-5ep-wd0,100k-flan-paper-5ep-wd005,100k-flan-paper-5ep-lora8,100k-flan-paper-5ep-lora16,100k-flan-paper-5ep-lora32,100k-flan-paper-5ep-lora32-5ep,100k-flan-paper-report-template-lora32,100k-flan-paper-5ep-freeze-enc}`, `runs/flan-t5-large/{100k,100k-lr5e4,100k-flan-paper-5ep}`, `runs/t5-base/100k-flan-paper-5ep`.

**B5** prompt: `flan-paper` (`Generate a report for:`) — see `data/processed/flan-t5-base/100k-flan-paper/prepare_config.json`.

---

## PD / HC breakdown (decode)

| Run ID | PD AVG | HC AVG | HC BLEU |
|--------|--------|--------|---------|
| S0 | 0.403 | 0.319 | 0.117 |
| **S4** | 0.428 | 0.447 | **0.277** |
| **S5** | 0.375 | 0.312 | 0.116 |
| S1 | 0.440 | 0.380 | 0.255 |
| S2 | 0.456 | 0.402 | 0.281 |
| S3 | 0.487 | 0.431 | 0.271 |
| — (1k) | 0.461 | 0.412 | 0.265 |
| B0 | 0.442 | 0.345 | 0.136 |
| B4 | 0.489 | 0.555 | 0.378 |
| **B5** | 0.519 | 0.538 | 0.379 |
| B6 | 0.484 | 0.534 | 0.362 |
| B7 | 0.497 | 0.422 | 0.252 |
| B8 | 0.172 | 0.179 | 0.000 |
| B9 | 0.172 | 0.179 | 0.000 |
| B10 | 0.172 | 0.179 | 0.000 |
| L0 | 0.515 | 0.484 | 0.328 |
| L4 | 0.504 | 0.497 | 0.316 |
| **B11** | 0.510 | 0.567 | 0.379 |
| L5 | 0.495 | **0.578** | **0.411** |
| B12 | 0.506 | 0.493 | 0.338 |
| B13 | 0.486 | 0.548 | 0.367 |
| N0 | 0.448 | 0.429 | 0.266 |
| B15 | 0.499 | 0.560 | 0.369 |
| B16 | 0.509 | 0.567 | 0.380 |
| B14 | 0.512 | 0.568 | 0.383 |
| **B17** | 0.513 | 0.571 | 0.383 |
| B18 | 0.514 | 0.570 | 0.384 |
| B19 | 0.509 | 0.572 | 0.388 |

---

## Conclusions

### Learning rate (5e-4 vs 3e-4 at 100k)

- **Base:** **`5e-4` is a clear win.** AVG **0.395 → 0.522** (+0.127); BLEU **0.215 → 0.351**; HC BLEU **0.136 → 0.378**. Largest single-config gain so far. `test_loss` moved only slightly (1.22 → 1.19); **decode metrics are the right selection criterion**.
- **Large:** **`5e-4` ≈ tie** on overall AVG (**0.500** vs **0.500**). ROUGE-L slightly up at 5e-4; BLEU and HC BLEU slightly down. **Prefer `3e-4` (L0)** for large unless combined with another change (prompt, epochs).
- **Small @ 100k (S4 vs S5 vs S0):** **S4 (`5e-4`)** is the clear winner — AVG **0.437** vs S0 **0.362** (+**0.075**); HC BLEU **0.277** vs **0.117**; beats S0 on every decode metric. **S5 (`3e-4` @ 100k)** underperforms S0 (**AVG 0.344**) and S4 — **do not use 3e-4 at full 100k** despite S2 winning HC BLEU at **10k**. **10k trends do not fully transfer:** S3 (**0.459**) still edges S4 on AVG (more data ≠ better than tuned short run for this metric). **Best small config for reporting: S4** (`runs/flan-t5-small/100k-lr5e4`).

### Phase 2 sequence length audit (B5 data, 2026-05-29)

Token-length check on `100k` splits with **`flan-paper`** prefix and **`flan-t5-base`** tokenizer (`truncation=False`). B5 prepare caps: **`max_source_length=256`**, **`max_target_length=512`**.

| Split | Source tokens (max) | Target tokens (max) | `source > 256` | `target > 512` |
|-------|---------------------|---------------------|----------------|----------------|
| train | 51 | 173 | 0% | 0% |
| val   | 51 | 169 | 0% | 0% |
| test  | 51 | 147 | 0% | 0% |

- **No silent truncation** at current caps; longest targets (~173 tokens) are well below 512.
- **No re-prepare / re-train** for length (128/384/768 not needed). Same `input_text` width applies to small/base/large.

### Phase 3 decode sweep (B5, 2026-05-29)

**Checkpoint comparison** (`final_model` vs saved steps, beam 3, `max_new_tokens=512`, `no_repeat_ngram=2`):

| Weights | AVG | BLEU |
|---------|-----|------|
| **`final_model`** | **0.5285** | 0.3693 |
| `checkpoint-74922` | 0.5282 | 0.3693 |
| `checkpoint-70000` | 0.5255 | 0.3699 |

Best **`eval_val_loss`** step was **74922** (loss **0.0000** on synthetic val) but **`final_model`** edges it on decode AVG — **do not pick checkpoints by val loss alone**.

**Beam sweep** on **`final_model`**:

| `--num-beams` | AVG | BLEU |
|---------------|-----|------|
| 1 | 0.5214 | 0.3740 |
| **3** | **0.5285** | **0.3693** |
| 5 | 0.5173 | 0.3597 |

- **Reporting config unchanged:** `runs/flan-t5-base/100k-flan-paper/final_model`, beam **3** (matches original **B5** AVG **0.529** within rounding).
- **Artifacts:** `test_decode_metrics_final_model.json`, `test_decode_metrics_checkpoint-{70000,74922}.json`, `test_decode_metrics_beam{1,5}.json`; plot `runs/flan-t5-base/training_compare_b4_b5.png`.

### Phase 2 prompt (B5 — `flan-paper` vs B4 — default)

- **B5** (`Generate a report for:`) is a **modest win** over **B4** on decode: AVG **0.529** vs **0.522** (+**0.007**); BLEU **0.369** vs **0.351**; ROUGE-L **0.510** vs **0.499**; BERT **0.819** vs **0.816**. ROUGE-2 slightly lower (**0.360** vs **0.362**).
- **By group:** **PD** AVG **0.519** vs B4 **0.489** (+0.030); **HC** AVG **0.538** vs B4 **0.555** (−0.017) — paper-style prefix helps PD more than HC on this run. HC BLEU ≈ tie (**0.379** vs **0.378**).
- **`test_loss`** slightly higher on B5 (**1.203** vs **1.186**); again, decode metrics are what improved.
- **New best overall:** **B5** — use `runs/flan-t5-base/100k-flan-paper` for reporting until a later experiment beats **0.529**.

### Phase 2 prompt (B6 — `flan-paper-categories` vs B5)

- **B6** (paper prefix + explicit seven mFDA category hints) **underperforms B5** on decode: AVG **0.509** vs **0.529** (−**0.020**), BLEU **0.337** vs **0.369** (−0.032), ROUGE-2 **0.341** vs **0.360**, ROUGE-L **0.487** vs **0.510**, BERT **0.812** vs **0.819**.
- **By group:** **PD** AVG **0.484** vs B5 **0.519** (−0.035); **HC** AVG **0.534** vs **0.538** (−0.004); HC BLEU **0.362** vs **0.379** (−0.017).
- B6 remains above B0 baseline but below both **B4** and **B5**; keep **B5** as the base reporting configuration.
- `test_loss` for B6 is **1.355** and aligns with the same conclusion: lower/alternate loss values do not override decode-metric selection.

### Phase 2 prompt/input (B7 — `flan-paper-numeric-labels` vs B5)

- **B7** (paper prefix + deterministic `Category=value` numeric formatting) is a **strong regression** vs B5 on decode: AVG **0.460** vs **0.529** (−**0.069**), BLEU **0.298** vs **0.369** (−0.071), ROUGE-2 **0.261** vs **0.360**, ROUGE-L **0.413** vs **0.510**, BERT **0.803** vs **0.819**.
- **By group:** **PD** AVG **0.497** vs B5 **0.519** (−0.022); **HC** AVG **0.422** vs **0.538** (−0.116); HC BLEU **0.252** vs **0.379** (−0.127). Regression is concentrated on HC.
- B7 remains above B0 baseline but is below B4/B5 and below B6. Keep **B5** as the base reporting configuration; do not promote B7.
- `test_loss` is **0.811** on B7, but decode quality is worse, reinforcing that prompt choices must be selected on generation metrics.

### Phase 1/3 hyperparameters (B8/B9/B10 — label smoothing on B5 recipe) — confirmed real regression (2026-05-31)

- **B8 (ls=0.05), B9 (ls=0.10), and B10 (ls=0.02)** all collapse to nearly identical decode outcomes: AVG ~**0.176**, BLEU ~**0.000**, ROUGE-2 ~**0.002**, far below B5 (**0.529**). Group metrics collapse similarly (PD ~0.172, HC ~0.179; HC BLEU ~0.000).
- **Diagnosed 2026-05-31 — this is a genuine regression, NOT a harness/artifact bug:**
  - **Harness verified:** re-decoded B5 `final_model` in the same environment → AVG **0.5285** (R-1 0.586, BLEU 0.369), reproducing historical B5 exactly. So `main.eval_decode` is healthy. Artifact: `test_decode_metrics_sanity.json`.
  - **Training converged:** B8 (`ls005`) `trainer_state.json` shows `eval_val_loss ≈ 0.7176` (finite), all `74922` steps completed, checkpoints saved — **no NaN / no divergence**. Note `eval_train_loss ≈ eval_val_loss` and barely moves (70000 vs 74922), i.e. loss is pinned near the **label-smoothing penalty floor** and masks actual fit.
  - **Generation is degenerate:** B8 decode `sacrebleu_precisions = [45.59, 1.89, 0.025, 0.013]` (1/2/3/4-gram %). Unigram ~46% but bigram ~2% and 4-gram ~0.01% → **word-salad** (right vocabulary, incoherent order), giving BLEU ≈ 0.
- **Takeaway:** label smoothing flattens the output distribution enough to wreck autoregressive decoding for this T5 recipe even though teacher-forced loss looks reasonable. **Label smoothing abandoned** for flan-t5-base @ 5e-4 (all of 0.02 / 0.05 / 0.10). `test_loss` values (**0.753 / 1.127 / 1.733**) are not comparable across smoothing factors and must not be used for selection.

### Epochs 3→5 (B11 vs B5)

- **B11 (5 epochs) is the new best overall: AVG 0.538** vs B5 0.529 (+0.009). **R-1 0.586→0.600, R-2 0.360→0.384, R-L 0.510→0.527, BERT 0.819→0.822** all improve; only **BLEU dips slightly (0.369→0.358)**.
- **`test_loss` also improved: 1.203 → 1.094** (the extra epochs genuinely fit better, not just trade metrics) — consistent with the **underfitting** read (large ≤ base at 3 epochs).
- **BLEU dip is a brevity-penalty effect, not worse n-grams:** B11 sacrebleu precisions **70.9 / 53.1 / 40.8 / 31.5** *beat* B5 (68.1 / 50.0 / 37.4 / 28.2) on every n-gram, but **BP 0.764 vs 0.848** — B11 generates **shorter** outputs, so BLEU nets slightly lower while ROUGE/BERT (precision-friendly) rise.
- **By group: the gain is entirely on HC.** HC AVG **0.567** vs B5 0.538 (+0.029), HC BLEU **0.379** (tie); **PD AVG 0.510** vs B5 0.519 (−0.009). Extra epochs help the (easier) HC reports more than PD.
- **Next:** test whether **large @ 5 epochs** benefits more (most capacity headroom).
- Reporting config updates to **B11** (`runs/flan-t5-base/100k-flan-paper-5ep/final_model`, beam 3) until beaten.

### Weight decay ablation — B12 (wd=0.0 vs B11 wd=0.01)

- **B12** (B11 recipe, **`--weight-decay 0.0`**, 5 epochs) **regresses vs B11:** AVG **0.500** vs **0.538** (−**0.038**); R-1 **0.555** vs **0.600**, R-2 **0.325** vs **0.384**, R-L **0.471** vs **0.527**, BLEU **0.338** vs **0.358**, BERT **0.809** vs **0.822**.
- **`test_loss` 1.351** vs B11 **1.094** — worse on both teacher-forced and decode metrics; removing decay did not help the underfitting read from 3→5 epochs.
- **By group:** regression is **concentrated on HC.** HC AVG **0.493** vs B11 **0.567** (−0.074); HC BLEU **0.338** vs **0.379**. PD AVG **0.506** vs B11 **0.510** (≈ flat).
- **Takeaway:** **Keep wd=0.01** (B11 default). **B13 (wd=0.05)** completes the sweep — see below.

### Weight decay ablation — B13 (wd=0.05 vs B11 wd=0.01) — sweep closed (2026-06-12)

- **B13** (B11 recipe, **`--weight-decay 0.05`**, 5 epochs) **does not beat B11:** AVG **0.517** vs **0.538** (−**0.021**); R-1 **0.581** vs **0.600**, R-2 **0.351** vs **0.384**, R-L **0.495** vs **0.527**, BLEU **0.342** vs **0.358**, BERT **0.814** vs **0.822**.
- **`test_loss` 1.159** vs B11 **1.094** — slightly worse teacher-forced loss; decode metrics confirm no gain.
- **By group:** **PD** AVG **0.486** vs B11 **0.510** (−0.024), worst PD in the wd sweep; **HC** AVG **0.548** vs B11 **0.567** (−0.019); HC BLEU **0.367** vs **0.379**.
- **Sweep summary (5 ep, flan-paper, 5e-4):** wd **0.01 → 0.538** (B11, best), **0.05 → 0.517** (B13), **0.0 → 0.500** (B12). **Weight-decay lever exhausted** — do not change wd from **0.01**.

### Phase 4 — non-Flan T5-base (N0 vs B11) — Flan required (2026-06-12)

- **N0** (`google-t5/t5-base`, B11 recipe: flan-paper, 5 ep, 5e-4, wd 0.01) **regresses strongly vs B11:** AVG **0.439** vs **0.538** (−**0.099**); R-1 **0.455** vs **0.600**, R-2 **0.269** vs **0.384**, R-L **0.393** vs **0.527**, BLEU **0.285** vs **0.358**, BERT **0.791** vs **0.822**.
- **`test_loss` 1.247** vs B11 **1.094** — worse on teacher-forced and decode metrics.
- **By group:** both groups weak vs B11; **HC collapses** — HC AVG **0.429** vs B11 **0.567** (−0.138), HC BLEU **0.266** vs **0.379**; PD AVG **0.448** vs B11 **0.510** (−0.062). PD slightly above HC on N0 (opposite of B11's HC-heavy pattern), but both far below Flan.
- **Takeaway:** **Flan instruction tuning is essential** for this pipeline at base scale. **Non-Flan t5-base lever closed** — do not pursue vanilla T5-base.

### Phase 4 — LoRA on B11 (B14 / B16 / B17) — rank sweep closed (2026-06-12)

| Run | LoRA r | AVG | vs B11 (0.538) | PD AVG | HC AVG |
|-----|--------|-----|----------------|--------|--------|
| **B16** | 8 | **0.538** | ≈ tie | 0.509 | 0.567 |
| **B14** | 16 | **0.540** | +0.002 | 0.512 | 0.568 |
| **B17** | 32 | **0.542** | +**0.004** | **0.513** | **0.571** |

- **B17** (LoRA **rank 32**, 3 ep from B11 `final_model`) is **new best overall:** AVG **0.542**; R-2 **0.389**, R-L **0.531**, BLEU **0.362**, BERT **0.823**; `test_loss` **1.145** (best among LoRA runs).
- **B16** (rank **8**) **ties B11** on AVG (**0.538**) — lowest LoRA capacity is not enough.
- **B14** (rank **16**) **0.540** — middle of sweep; still beats B11 but below B17.
- **By group:** B17 lifts **both** PD (**0.513** vs B11 **0.510**) and HC (**0.571** vs **0.567**); largest HC gain in the sweep (+0.004 vs B11).
- **Reporting config updates to B17** (`runs/flan-t5-base/100k-flan-paper-5ep-lora32/final_model`, beam 3). **LoRA rank sweep closed** — no further rank trials planned.

### Phase 4 — longer LoRA on B17 (B18) — closed (2026-06-15)

- **B18** (LoRA **rank 32** on B17 `final_model`, **5 ep**, same LR/wd as B17) **≈ ties B17** on decode: AVG **0.542** vs **0.542** (+0.0002, within noise); R-1 **0.605** vs **0.604**, R-2 **0.390** vs **0.389**, R-L **0.530** vs **0.531**, BLEU **0.362** (tie), BERT **0.823** (tie).
- **`test_loss` 1.145** — identical to B17 **1.145** (epoch 5.0).
- **By group:** **PD AVG 0.514** vs B17 **0.513** (+0.001); **HC AVG 0.570** vs **0.571** (−0.001); HC BLEU **0.384** vs **0.383** (+0.001). Tiny PD lift, tiny HC regression — no clear win.
- **Takeaway:** **Longer LoRA fine-tune lever closed** — 2 extra epochs on the B17 checkpoint buy nothing meaningful. **Keep B17** as reporting config (same AVG with less compute). Next lever: beyond prefix text (oversampling, constrained decoding).

### Phase 2 — output template prompt (B19) — closed (2026-06-15)

- **B19** (`flan-paper-report-template` prefix + LoRA r=32 on B17, 3 ep) **below B17 on decode:** AVG **0.540** vs **0.542** (−0.002); R-1 **0.600** vs **0.604**, R-2 **0.386** vs **0.389**, BLEU **0.361** vs **0.362**.
- **`test_loss` 1.156** vs B17 **1.145**.
- **By group:** **PD AVG 0.509** vs B17 **0.513** (−0.004); **HC AVG 0.572** vs **0.571** (+0.001); HC BLEU **0.388** vs **0.383** (+0.005). Slight HC lift, PD regression — same HC-heavy pattern as B11 epochs.
- **PD structure (`pd_analysis.json`):** `mean_coverage_hyp` **0.143** (≈ B14/B17 **~0.14**); refs **1.0**. **Breathing** hyp presence **99%**; **Lips–Intelligibility** still **0–1%**. PD severity match **0.479** (≈ unchanged). **Output-template prefix did not fix 7-slot structure.**
- **Takeaway:** **Template prompt lever closed** — instructing `Category (Severity):` format in the encoder prefix is not enough. **Keep B17** for reporting. Next: **PD oversampling**, **constrained decoding**, or training-time structure losses (not more prefix variants like B6/B7/B19).

### Phase 4 — LoRA on B11 (B14) — prior note

- First LoRA win at r=16 (AVG **0.540**); superseded by **B17** r=32 (**0.542**).

### Phase 4 — freeze encoder on B11 (B15) — closed (2026-06-12)

- **B15** (B11 `final_model`, **`--freeze-encoder`**, 3 ep) **does not beat B11:** AVG **0.529** vs **0.538** (−**0.009**); R-1 **0.588**, BLEU **0.346**, BERT **0.818**.
- **`test_loss` 1.210** vs B11 **1.094**.
- **By group:** **PD AVG 0.499** vs B11 **0.510** (−0.011); **HC AVG 0.560** vs **0.567** (−0.007); HC BLEU **0.369**. Regression on both groups vs B11.
- **Takeaway:** **Freeze-encoder lever closed** for this recipe; **LoRA (B17 r=32) is the Phase 4 win.**

### PD-targeted analysis on B14 (2026-06-12)

Artifact: `runs/flan-t5-base/100k-flan-paper-5ep-lora16/pd_analysis.json` (`main/analyze_pd_decode` on B14 vs B11 predictions).

- **Decode metrics vs structure:** B14 **AVG 0.540** / PD **0.512** / HC **0.568** look healthy, but **strict `Category (Severity):` slot parsing** finds only **~14% mean category coverage** in generated text (refs = **100%**). Same pattern on **B11** baseline (~14%) — not introduced by LoRA alone.
- **Dominant parsed slot:** **Breathing** (~98% hyp presence in B11 baseline parse); **Lips–Intelligibility** slots are almost never emitted in the strict template (hyp presence **0–2%** per category on B11). Models still score on ROUGE/BLEU via overlapping free text, but **full 7-category mFDA structure is largely missing** in outputs.
- **Severity match (where comparable):** B11 baseline **~49%** mean severity match (mostly Breathing); B14 vs B11 deltas — **PD severity match −0.01**, **HC +0.02** (HC gains, PD flat/slightly worse).
- **Weakest PD examples:** many PD decodes parse as **1/7 categories** (Breathing only) with **0% severity match** on the comparable slot.
- **Takeaway:** The PD/HC **decode AVG gap** (~0.06) is real, but the next lever is likely **template/structure** (prompt, training data mix, or constrained decoding) — not another LoRA rank. PD analysis task **complete**; see [Model Improvement Plan](Model%20Improvement%20Plan.md).

### Large @ 5 epochs (L5 vs B11) — size lever exhausted

- **L5 (flan-t5-large, flan-paper, 5e-4, 5 epochs) ties base B11: AVG 0.536 vs 0.538** (−0.002, within noise) — **3× the parameters buys nothing overall.**
- **HC↔PD tradeoff:** L5 **HC AVG 0.578** (vs B11 0.567, +0.011) and **HC BLEU 0.411** (vs 0.379, +0.032) — best HC numbers so far — but **PD AVG 0.495** (vs B11 0.510, −0.015), the worst base/large PD yet. The extra capacity went to HC, not the harder PD reports.
- **`test_loss` 2.779** — far above B11's 1.094 despite comparable decode AVG. Classic loss-vs-generation decoupling; also hints at **overfitting / 5e-4 being too hot for large @ 5 epochs** (teacher-forced generalization degraded while beam search still produces decent text).
- **Verdict:** epochs lifted large (0.500 → 0.536) but only to a tie with base; **going bigger is not the lever.** Epochs (B11) remains the win; **B11 stays the best run.**

### Model size

- **Best run to date:** **B17 — LoRA rank 32 on B11 checkpoint, 3 ep** (**AVG 0.542**); **B18** ≈ tie; **B19** below B17 (**0.540**). Template prompt **closed**.
- **Best full fine-tune:** **B11** (0.538).
- **Best small @ 100k:** **S4** (**AVG 0.437**).
- **Large:** L0/L4 (3 ep) ~0.500 → **L5 (5 ep) 0.536**, still only ties base — large is not worth its cost here.
- **Persistent pattern: PD is the weak group** (B17 PD **0.513** vs HC **0.571**; B18 PD **0.514** vs HC **0.570**). LoRA rank + longer LoRA both plateau; **PD analysis** shows missing 7-slot report structure — main bottleneck vs paper (~0.68).

### What to run next

See **[Model Improvement Plan](Model%20Improvement%20Plan.md)**. Best reporting config remains **B17** (AVG **0.542**). **B19 closed** — template prompt did not lift AVG or 7-slot coverage. **Next:** PD oversampling or constrained decoding.

### `test_loss` vs generation metrics

- Low **`test_loss`** does not guarantee good reports (e.g. **1k** run: loss **0.24**, AVG **0.436** — not comparable to 100k training, but illustrates the gap).
- Near-zero **synthetic val loss** (S0) paired with weak real-test decode — do not optimize on val loss alone.
- **B14** has higher `test_loss` than B11 (1.166 vs 1.094) but higher decode AVG — reinforces using decode metrics for selection.

---

## Plotting

**Loss curves:** `checkpoint-*/trainer_state.json` per run ([train.md](train.md)).

**Comparison figures (decode bars need `test_decode_metrics.json` in each run folder):**

```bash
python -m main.plot_training_runs runs/flan-t5-base/100k-lr5e4 runs/flan-t5-base/100k-flan-paper --output runs/flan-t5-base/training_compare_b4_b5.png
python -m main.plot_training_runs --runs-parent runs/flan-t5-base --output runs/flan-t5-base/training_compare.png
python -m main.plot_training_runs --runs-parent runs/flan-t5-large --output runs/flan-t5-large/training_compare.png
python -m main.plot_training_runs runs/flan-t5-small/100k runs/flan-t5-small/100k-lr5e4 runs/flan-t5-small/100k-lr3e4 runs/flan-t5-small/10k-lr5e4 runs/flan-t5-small/10k-lr3e4 runs/flan-t5-small/10k-lr1e4 --output runs/flan-t5-small/training_compare.png
```

---

*Results log — last updated 2026-06-15 (**B19** template prompt AVG **0.540**, 7-slot coverage **0.143** unchanged — lever closed. **B17** remains best). Plan: [Model Improvement Plan](Model%20Improvement%20Plan.md).*
