# Training — `main/train.py`

[`main/train.py`](../main/train.py) is the minimal **Flan-T5 / T5** fine-tuning entry point. It loads a tokenized Hugging Face `DatasetDict` from disk (produced by [`main/prepare.py`](../main/prepare.py) with `--tokenize`), runs **`Seq2SeqTrainer`**, and exports a final checkpoint plus tokenizer.

For the upstream pipeline (splits, prompts, tokenization), see [data-pipeline.md](data-pipeline.md).

For the improvement checklist (all Flan-T5 sizes), see [Model Improvement Plan.md](Model%20Improvement%20Plan.md). For logged decode metrics and experiment conclusions, see [training_progress.md](training_progress.md). For generation metrics after training, see [eval-decode.md](eval-decode.md). For TinyGPU login, queue, interactive GPU, and `scp`, see [hpc-commands.md](hpc-commands.md).

## Prerequisites

- Conda env **`AcousticDrivenGeneration`** (or equivalent) with **`torch`**, **`transformers`**, **`datasets`**, **`accelerate>=1.1`**, and **`peft>=0.10`** for Phase 4 LoRA (see [`requirements.txt`](../requirements.txt)).
- A **`DatasetDict` on disk** from prepare **with `--tokenize`**. By default that lives at **`<processed_dir>/tokenized/`** unless you set `--save-tokenized-dir`.

  ```bash
  python -m main.prepare --output-dir <processed_dir> --train-size <1k|10k|100k> --tokenize
  ```

  Optionally set an explicit path: `--save-tokenized-dir <tokenized_dir>`. Use the same Hugging Face model for tokenization as for training (defaults below).

## Input data contract

`--tokenized-dir` must point at a folder that contains **`dataset_dict.json`** (Hugging Face `DatasetDict.save_to_disk`), or at the **parent** prepare output directory if it contains a **`tokenized/`** subfolder with that file. It must include at least:

| split | required |
| --- | --- |
| `train` | yes |
| `val`   | yes |

Each split is expected to include the usual seq2seq fields:

- **`input_ids`**, **`attention_mask`** — encoder inputs  
- **`labels`** — decoder targets with padding masked as **`-100`** (as produced by [`main/tokenization.py`](../main/tokenization.py))

Optional columns (for example `sample_id`, `example_hash`) may be present; the trainer drops unused columns before the forward pass.

**Important:** `--model-name` should match the tokenizer used when building the dataset (`--tokenizer-model` in `main.prepare`, default **`google/flan-t5-small`**). Mismatched vocabularies will corrupt training.

### Standard T5 and Flan-T5 checkpoints

`--model-name` is any Hugging Face id that loads with `AutoModelForSeq2SeqLM` (this project targets **T5 v1.1–style** encoder–decoder checkpoints). The usual public sizes are below.

**Parameter totals** are **rounded** from the published **`model.safetensors`** weight tensors (total float elements in the checkpoint). Small differences can appear across formats or library versions.

| Hub id | Parameters (≈) |
| --- | ---: |
| `google-t5/t5-small` | 61M |
| `google-t5/t5-base` | 223M |
| `google-t5/t5-large` | 738M |
| `google-t5/t5-3b` | 2.85B |
| `google-t5/t5-11b` | 11B |
| `google/flan-t5-small` | 77M |
| `google/flan-t5-base` | 248M |
| `google/flan-t5-large` | 783M |
| `google/flan-t5-xl` | 2.85B |
| `google/flan-t5-xxl` | 11.3B |

**Naming:** the **`google-t5/`** org hosts the maintained **original T5** mirrors; older tutorials sometimes wrote **`google/t5-small`** etc., which may **not** resolve on the Hub (use **`google-t5/t5-base`**, not `google/t5-base`, when downloading). **Flan-T5** lives under **`google/flan-t5-*`**: same **architecture sizes** as T5 at each tier, but instruction-tuned weights. Within a **size tier** (small vs base vs …), T5 and Flan-T5 share the **same tokenizer vocabulary**, so keep **`--tokenizer-model`** and **`--model-name`** on the **same tier** (and rebuild the tokenized dataset if you change tier).

**Flan vs non-Flan (N0):** at base scale, **`google-t5/t5-base`** with the B11 recipe reached decode **AVG 0.439** vs **B11 (flan-t5-base) 0.538** — see [training_progress.md](training_progress.md). **Use Flan-T5 for this pipeline**; non-Flan base is closed.

**Related but different vocab:** **mT5** (`google/mt5-small`, `google/mt5-base`, …) is multilingual T5 with a **different SentencePiece vocabulary**. Do not mix mT5 tokenization with T5/Flan-T5 checkpoints unless you deliberately re-tokenize with the matching **`google/mt5-*`** tokenizer.

**Compute:** **XL / XXL / 3B / 11B** need large GPU memory (or multi-GPU / CPU with very long runtimes). Defaults in this repo assume **small**–**base**–scale runs.

## Recommended base recipe (B11 — current best)

As of the runs logged in [training_progress.md](training_progress.md), the best **flan-t5-base** config is **B11**:

| Setting | Value |
| --- | --- |
| Prepare | `data/processed/flan-t5-base/100k-flan-paper` (`--prompt-style flan-paper`) |
| Tokenized dir | `data/processed/flan-t5-base/100k-flan-paper/tokenized` |
| Model | `$WORK/models/flan-t5-base` (or `google/flan-t5-base`) |
| Epochs | **5** |
| Learning rate | **5e-4** |
| Weight decay | **0.01** |
| Label smoothing | **0.0** (do **not** use — B8/B9/B10 collapsed decode; see training_progress) |
| Batch (train / eval) | 4 / 4 per device |
| Warmup ratio | 0.1 |
| eval / save steps | 5000 / 5000 |
| Output | `runs/flan-t5-base/100k-flan-paper-5ep` |
| Decode **AVG** | **0.538** (beam 3; `main.eval_decode`) |

Slurm: [`scripts/hpc/train_flan_t5_base_100k_flan_paper_5ep_a100.slurm`](../scripts/hpc/train_flan_t5_base_100k_flan_paper_5ep_a100.slurm).

**Selection rule:** pick configs by **decode AVG** from `test_decode_metrics.json`, not by `test_loss` or synthetic val loss alone.

### Closed levers (do not re-run without new hypothesis)

| Lever | Result |
| --- | --- |
| Model size (large @ 5 ep, L5) | Ties B11 (AVG 0.536) — not worth 3× cost |
| Weight decay (B12 wd=0.0, B13 wd=0.05) | Both regress vs B11 — keep **0.01** |
| Label smoothing (B8–B10) | Degenerate generation — **do not use** |
| Prompt variants (B6/B7) | Below B5/B11 |
| Non-Flan t5-base (N0) | AVG 0.439 — Flan required |

## Phase 4 — next training experiments

Phase 4 flags are implemented in [`main/train.py`](../main/train.py) (requires **`peft`** in the conda env). Start from **B11** weights by setting **`--model-name`** to `runs/flan-t5-base/100k-flan-paper-5ep/final_model` (or `$WORK/models/flan-t5-base` for from-scratch baselines). **`--lora-rank` and `--freeze-encoder` are mutually exclusive.**

| Mode | Flags | Notes |
| --- | --- | --- |
| **LoRA** | `--lora-rank 16` (optional `--lora-alpha`, `--lora-dropout`) | Adapters merged into **`final_model/`** on save so **`main.eval_decode`** works unchanged. |
| **Freeze encoder** | `--freeze-encoder` | Trains decoder only; full decoder weights updated. |

Suggested experiment IDs (log in [training_progress.md](training_progress.md)):

- **B14** — LoRA on B11 checkpoint, rank 16, 3 epochs, same LR/wd as B11  
- **B15** — freeze encoder on B11 checkpoint, 3–5 epochs, same LR/wd as B11  
- **B18** — LoRA rank 32 on **B17** `final_model`, **5 epochs** — AVG **0.542** (≈ tie B17); **closed**
- **B19** — `flan-paper-report-template` + LoRA r=32 on B17, 3 ep — AVG **0.540** (below B17); category coverage **≈14%** unchanged; **closed**

**LoRA example (B14-style):**

```bash
python -m main.train \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --output-dir runs/flan-t5-base/100k-flan-paper-5ep-lora16 \
  --model-name runs/flan-t5-base/100k-flan-paper-5ep/final_model \
  --lora-rank 16 \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --learning-rate 5e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --logging-steps 100 \
  --eval-steps 5000 \
  --save-steps 5000 \
  --save-total-limit 2 \
  --seed 42 \
  --require-gpu
```

**Freeze-encoder example (B15-style):**

```bash
python -m main.train \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --output-dir runs/flan-t5-base/100k-flan-paper-5ep-freeze-enc \
  --model-name runs/flan-t5-base/100k-flan-paper-5ep/final_model \
  --freeze-encoder \
  --num-train-epochs 3 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --learning-rate 5e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --logging-steps 100 \
  --eval-steps 5000 \
  --save-steps 5000 \
  --save-total-limit 2 \
  --seed 42 \
  --require-gpu
```

After training, run **`main.eval_decode`** on **`final_model/`** (see [eval-decode.md](eval-decode.md)). Compare decode **AVG** to **B11 (0.538)**.

Until results are logged, continue **PD analysis** (group balance, decode PD/HC tables, plots) — see [Model Improvement Plan.md](Model%20Improvement%20Plan.md).

## Model inputs, outputs, validation, and loss

### Inputs (features → tokens)

Upstream CSVs provide an **`Instructions`** field; [`etl/etl_lib.py`](../etl/etl_lib.py) parses **seven numeric biomarkers** in fixed order: **Breathing, Lips, Palate, Larynx, Monotonicity, Tongue, Intelligibility**. They are formatted into a single string **`input_text`** (e.g. `Breathing: … Lips: …` with stable spacing and precision).

For T5-style models, [`main/prompts.py`](../main/prompts.py) prepends a task prefix to build **`source_text`** (see `--prompt-style` in `main.prepare`: **`default`** = mFDA biomarker instruction; **`flan-paper`** = `Generate a report for:`; **`flan-paper-categories`** = paper prefix + explicit seven mFDA category hints; **`flan-paper-numeric-labels`** = paper prefix + deterministic `Category=value` formatting for the seven biomarkers; **`flan-paper-report-template`** = seven-slot `Category (Severity):` output template + biomarkers (B19)). [`main/tokenization.py`](../main/tokenization.py) tokenizes that into **`input_ids`** / **`attention_mask`** for the **encoder**.

So the model does **not** consume a separate continuous feature tensor; it sees **text derived from the seven numbers** plus the prefix.

### Outputs (what is predicted)

The target is the **clinical report** string: **`target_text`**, aligned with the paper’s mFDA-style report text (same column as normalized **`report_raw`** in the ETL pipeline). It is tokenized into decoder **`labels`**. Padding tokens in **`labels`** are replaced by **`-100`** so they are ignored in the loss.

### Validation and splits (not k-fold CV)

The repository uses **fixed files**, not rotating k-fold cross-validation:

- **`train`**: synthetic prompt–report pairs (1k / 10k / 100k subsets), loaded via [`main/io.py`](../main/io.py).
- **`val`**: synthetic validation split (`val_split.csv`).
- **`test`**: real reports (`test_split.csv`). If this split is present in the tokenized `DatasetDict`, **`main/train.py`** runs a final **`evaluate()`** on it **after** training (on the best checkpoint loaded by the trainer) and writes **`<output_dir>/test_eval.json`** (metrics prefixed with **`test_`**, e.g. **`test_loss`**). That is **cross-entropy / NLL** like validation, not BLEU/ROUGE unless you extend the trainer.

During training, **`Seq2SeqTrainer`** evaluates on the **`val`** split every **`--eval-steps`** steps. With **`load_best_model_at_end=True`**, the best checkpoint is restored before the test pass and before **`final_model/`** is saved. Optional **`--eval-train`** adds periodic train-split eval for overfitting checks.

### Loss (cost function)

Fine-tuning uses the **default seq2seq objective** from Hugging Face: **token-level cross-entropy** (negative log-likelihood) on the decoder targets with **teacher forcing**, summed/averaged over non-padding positions (padding masked via **`-100`**). You can optionally apply label smoothing via **`--label-smoothing`** (maps to `label_smoothing_factor` in `Seq2SeqTrainingArguments`; default `0.0`). **On flan-t5-base @ 5e-4, label smoothing wrecked decode** even when loss looked fine — keep **`0.0`** for the B11 recipe.

**`--weight-decay`** applies **AdamW weight decay** on parameters; that is **regularization**, not the sequence prediction loss. **Best base value: `0.01`** (B11); `0.0` and `0.05` both regressed (B12/B13).

## What the script does

1. **`load_from_disk`** on `--tokenized-dir`.
2. Optionally truncates train/val/test with **`--max-train-samples`**, **`--max-eval-samples`**, **`--max-test-samples`** (debugging).
3. Loads **`AutoTokenizer`** and **`AutoModelForSeq2SeqLM`** from **`--model-name`**.
4. Builds **`Seq2SeqTrainingArguments`** with step-based **eval** and **save**, **`load_best_model_at_end=True`**, best metric **`eval_val_loss`** or **`eval_loss`** depending on **`--eval-train`** (lower is better).
5. If **`--max-steps`** is **> 0**, **`eval_steps`** and **`save_steps`** are clamped to at most **`max_steps`** (and at least **1**) so short runs still evaluate and checkpoint.
6. **`trainer.train()`**, then if a **`test`** split exists and **`--skip-test-eval`** is not set, **`trainer.evaluate(test, metric_key_prefix='test')`** and write **`test_eval.json`**.
7. **`save_model`** and **`tokenizer.save_pretrained`** under **`<output_dir>/final_model/`**.

Intermediate checkpoints live directly under **`--output-dir`** (subject to **`--save-total-limit`**).

## CLI reference

| argument | default | description |
| --- | --- | --- |
| `--tokenized-dir` | *(required)* | Path to `DatasetDict` on disk. |
| `--output-dir` | *(required)* | Run directory (checkpoints + `final_model/`). |
| `--model-name` | `google/flan-t5-small` | HF hub id; must match prepare tokenizer. |
| `--num-train-epochs` | `3.0` | Ignored if `--max-steps` > 0. |
| `--max-steps` | `-1` | If > 0, caps total optimizer steps and overrides epochs. |
| `--per-device-train-batch-size` | `4` | |
| `--per-device-eval-batch-size` | `8` | |
| `--gradient-accumulation-steps` | `1` | |
| `--learning-rate` | `3e-4` | |
| `--weight-decay` | `0.01` | |
| `--label-smoothing` | `0.0` | Label smoothing factor in `[0.0, 1.0)`; `0.0` disables smoothing. |
| `--warmup-ratio` | `0.1` | Warmup as a fraction of total training steps. |
| `--logging-steps` | `100` | |
| `--eval-steps` | `500` | Eval every N steps (clamped when `--max-steps` set). |
| `--save-steps` | `500` | Save every N steps (clamped when `--max-steps` set). |
| `--save-total-limit` | `2` | Rolling checkpoint cap. |
| `--resume-from-checkpoint` | *(none)* | Resume from `checkpoint-*` in `--output-dir` (`auto` / path). |
| `--seed` | `42` | |
| `--max-train-samples` | *(none)* | Use only the first N training examples. |
| `--max-eval-samples` | *(none)* | Use only the first N validation examples. |
| `--eval-train` | on | Also eval train split each eval step (`--no-eval-train` to disable). |
| `--eval-train-max-samples` | `5000` | Cap train rows for that periodic train eval; **`0`** = full train set. |
| `--max-test-samples` | *(none)* | Cap test rows for post-train **`test_eval.json`** only. |
| `--skip-test-eval` | off | Skip final test split evaluation even if `test` exists. |
| `--fp16` | off | Enables fp16 when **CUDA** is available; otherwise ignored. |
| `--bf16` | off | Enables bf16 when **CUDA or MPS** is available. |
| `--report-to` | `none` | `none`, `wandb`, or `tensorboard`. |
| `--predict-with-generate` | off | Runs generation during eval (slower); no extra metrics unless you extend the trainer. |
| `--require-gpu` | off | Exit if `torch.cuda.is_available()` is false (catches CPU-only PyTorch). |
| `--cpu-only` | off | Force CPU training; do not combine with `--require-gpu`. |
| `--lora-rank` | `0` | If **> 0**, apply PEFT LoRA (rank *r*); merged into `final_model/` on save. |
| `--lora-alpha` | *(2 × rank)* | LoRA scaling alpha. |
| `--lora-dropout` | `0.05` | LoRA dropout. |
| `--freeze-encoder` | off | Train decoder only; mutually exclusive with `--lora-rank` > 0. |

## Example commands

**B11 (best base recipe — after prepare with `--prompt-style flan-paper`):**

```bash
python -m main.train \
  --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
  --output-dir runs/flan-t5-base/100k-flan-paper-5ep \
  --model-name google/flan-t5-base \
  --num-train-epochs 5 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --learning-rate 5e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --logging-steps 100 \
  --eval-steps 5000 \
  --save-steps 5000 \
  --save-total-limit 2 \
  --seed 42 \
  --require-gpu
```

On TinyGPU, prefer the Slurm script: `sbatch.tinygpu scripts/hpc/train_flan_t5_base_100k_flan_paper_5ep_a100.slurm` (see [hpc-commands.md](hpc-commands.md)).

Full run (after `prepare --tokenize`; default tokenized path is under the same output dir):

```bash
python -m main.train --tokenized-dir data/processed/1k/tokenized --output-dir data/runs/flan-t5-small-baseline
```

You can also pass **`--tokenized-dir data/processed/1k`** if that folder contains **`tokenized/`** with the dataset.

Quick debug (few steps, small subsets):

```bash
python -m main.train --tokenized-dir data/processed_smoke/tokenized --output-dir data/runs/smoke --max-steps 5 --max-train-samples 64 --max-eval-samples 64 --logging-steps 1
```

## Outputs

| path | contents |
| --- | --- |
| `<output_dir>/` | Step checkpoints, trainer state (per Transformers). |
| `<output_dir>/test_eval.json` | Present when the tokenized dataset includes **`test`** and **`--skip-test-eval`** is off: **`test_loss`** and related HF eval metrics on real reports. |
| `<output_dir>/final_model/` | Best-effort final export: model weights + tokenizer for inference or further fine-tuning. |

## Notes and limitations

- **`test`** split: used **once** after training for **`test_eval.json`** (teacher-forcing loss). **Decode metrics** (BLEU / ROUGE / BERTScore / **AVG**) come from **`main.eval_decode`** → `test_decode_metrics.json` — run on a **GPU compute node** with **`--require-gpu`**, not on the `tinyx` login node (CPU decode times out). See [eval-decode.md](eval-decode.md).
- **`--predict-with-generate`**: enables generation in eval but **does not** attach BLEU/ROUGE/BERTScore; add `compute_metrics` in code when those metrics are wired up.
- **V100 / TinyGPU:** avoid **`--fp16`** on V100 for this project (historical NaN / bogus loss); A100 FP32 (no flag) is stable.
- Transformers may emit warnings (for example deprecation of `warmup_ratio`, tied weights, or hub cache symlinks on Windows); they do not change the documented CLI behavior unless you upgrade libraries and APIs diverge.
