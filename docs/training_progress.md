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

↓ lower is better for `test_loss`; ↑ higher is better for decode metrics.

**Run directories:** `runs/flan-t5-small/{100k,100k-lr5e4,100k-lr3e4,10k-lr1e4,10k-lr3e4,10k-lr5e4,1k}`, `runs/flan-t5-base/{100k,100k-lr5e4,100k-flan-paper,100k-flan-paper-categories,100k-flan-paper-numeric-labels,100k-flan-paper-ls002,100k-flan-paper-ls005,100k-flan-paper-ls010}`, `runs/flan-t5-large/{100k,100k-lr5e4}`.

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

### Model size

- **Best run to date:** **B5 — flan-t5-base @ 5e-4, flan-paper prompt** (**AVG 0.529**).
- **Best small @ 100k:** **S4** (**AVG 0.437**).
- **Large (L0/L4)** ~**0.500** — below **B5**.

### `test_loss` vs generation metrics

- Low **`test_loss`** does not guarantee good reports (e.g. **1k** run: loss **0.24**, AVG **0.436** — not comparable to 100k training, but illustrates the gap).
- Near-zero **synthetic val loss** (S0) paired with weak real-test decode — do not optimize on val loss alone.

### What to run next

See **[Model Improvement Plan](Model%20Improvement%20Plan.md)**. B8/B9/B10 label-smoothing collapse is now **diagnosed as a real regression** (harness verified via B5 sanity decode = 0.529; training converged; generation degenerate) — **label smoothing is closed**. **Next:** **B11 = B5 recipe @ 5 epochs** (base, flan-paper, 5e-4, no label smoothing) to test the untouched **epochs** lever — base `test_loss` ~1.20 and large (~0.500) underperforming base (0.529) both point to underfitting. If B11 helps, also try large @ 5 epochs, then a weight-decay-only ablation on the winner.

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

*Results log — last updated 2026-05-31 (B8/B9/B10 label-smoothing collapse diagnosed as real regression via B5 sanity decode; label smoothing closed, B11 5-epoch run queued). Plan: [Model Improvement Plan](Model%20Improvement%20Plan.md).*
