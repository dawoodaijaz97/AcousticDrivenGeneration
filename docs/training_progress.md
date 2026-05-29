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
| L0 | flan-t5-large | 100k | 3e-4 | 0.997 | 0.566 | 0.318 | 0.461 | 0.346 | 0.810 | **0.500** |
| L4 | flan-t5-large | 100k | 5e-4 | 1.057 | 0.568 | 0.314 | 0.482 | 0.326 | 0.812 | **0.500** |

↓ lower is better for `test_loss`; ↑ higher is better for decode metrics.

**Run directories:** `runs/flan-t5-small/{100k,100k-lr5e4,100k-lr3e4,10k-lr1e4,10k-lr3e4,10k-lr5e4,1k}`, `runs/flan-t5-base/{100k,100k-lr5e4,100k-flan-paper}`, `runs/flan-t5-large/{100k,100k-lr5e4}`.

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
| L0 | 0.515 | 0.484 | 0.328 |
| L4 | 0.504 | 0.497 | 0.316 |

---

## Conclusions

### Learning rate (5e-4 vs 3e-4 at 100k)

- **Base:** **`5e-4` is a clear win.** AVG **0.395 → 0.522** (+0.127); BLEU **0.215 → 0.351**; HC BLEU **0.136 → 0.378**. Largest single-config gain so far. `test_loss` moved only slightly (1.22 → 1.19); **decode metrics are the right selection criterion**.
- **Large:** **`5e-4` ≈ tie** on overall AVG (**0.500** vs **0.500**). ROUGE-L slightly up at 5e-4; BLEU and HC BLEU slightly down. **Prefer `3e-4` (L0)** for large unless combined with another change (prompt, epochs).
- **Small @ 100k (S4 vs S5 vs S0):** **S4 (`5e-4`)** is the clear winner — AVG **0.437** vs S0 **0.362** (+**0.075**); HC BLEU **0.277** vs **0.117**; beats S0 on every decode metric. **S5 (`3e-4` @ 100k)** underperforms S0 (**AVG 0.344**) and S4 — **do not use 3e-4 at full 100k** despite S2 winning HC BLEU at **10k**. **10k trends do not fully transfer:** S3 (**0.459**) still edges S4 on AVG (more data ≠ better than tuned short run for this metric). **Best small config for reporting: S4** (`runs/flan-t5-small/100k-lr5e4`).

### Phase 2 prompt (B5 — `flan-paper` vs B4 — default)

- **B5** (`Generate a report for:`) is a **modest win** over **B4** on decode: AVG **0.529** vs **0.522** (+**0.007**); BLEU **0.369** vs **0.351**; ROUGE-L **0.510** vs **0.499**; BERT **0.819** vs **0.816**. ROUGE-2 slightly lower (**0.360** vs **0.362**).
- **By group:** **PD** AVG **0.519** vs B4 **0.489** (+0.030); **HC** AVG **0.538** vs B4 **0.555** (−0.017) — paper-style prefix helps PD more than HC on this run. HC BLEU ≈ tie (**0.379** vs **0.378**).
- **`test_loss`** slightly higher on B5 (**1.203** vs **1.186**); again, decode metrics are what improved.
- **New best overall:** **B5** — use `runs/flan-t5-base/100k-flan-paper` for reporting until a later experiment beats **0.529**.

### Model size

- **Best run to date:** **B5 — flan-t5-base @ 5e-4, flan-paper prompt** (**AVG 0.529**).
- **Best small @ 100k:** **S4** (**AVG 0.437**).
- **Large (L0/L4)** ~**0.500** — below **B5**.

### `test_loss` vs generation metrics

- Low **`test_loss`** does not guarantee good reports (e.g. **1k** run: loss **0.24**, AVG **0.436** — not comparable to 100k training, but illustrates the gap).
- Near-zero **synthetic val loss** (S0) paired with weak real-test decode — do not optimize on val loss alone.

### What to run next

See **[Model Improvement Plan](Model%20Improvement%20Plan.md)**. **B5** done — optional **max_source_length** audit, second prompt variant, or **beam/checkpoint sweep** on **B5**. Consider **flan-paper** on **S4** only if you need a stronger small model.

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

*Results log — last updated from `runs/**/test_*.json`. Plan: [Model Improvement Plan](Model%20Improvement%20Plan.md).*
