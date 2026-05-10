# Training — `main/train.py`

[`main/train.py`](../main/train.py) is the minimal **Flan-T5 / T5** fine-tuning entry point. It loads a tokenized Hugging Face `DatasetDict` from disk (produced by [`main/prepare.py`](../main/prepare.py) with `--tokenize`), runs **`Seq2SeqTrainer`**, and exports a final checkpoint plus tokenizer.

For the upstream pipeline (splits, prompts, tokenization), see [data-pipeline.md](data-pipeline.md).

## Prerequisites

- Conda env **`AcousticDrivenGeneration`** (or equivalent) with **`torch`**, **`transformers`**, **`datasets`**, and **`accelerate>=1.1`** (see [`requirements.txt`](../requirements.txt)).
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

## Model inputs, outputs, validation, and loss

### Inputs (features → tokens)

Upstream CSVs provide an **`Instructions`** field; [`etl/etl_lib.py`](../etl/etl_lib.py) parses **seven numeric biomarkers** in fixed order: **Breathing, Lips, Palate, Larynx, Monotonicity, Tongue, Intelligibility**. They are formatted into a single string **`input_text`** (e.g. `Breathing: … Lips: …` with stable spacing and precision).

For T5-style models, [`main/prompts.py`](../main/prompts.py) prepends the task prefix (**`generate mFDA clinical speech report from acoustic biomarkers:`**) to build **`source_text`**. [`main/tokenization.py`](../main/tokenization.py) tokenizes that into **`input_ids`** / **`attention_mask`** for the **encoder**.

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

Fine-tuning uses the **default seq2seq objective** from Hugging Face: **token-level cross-entropy** (negative log-likelihood) on the decoder targets with **teacher forcing**, summed/averaged over non-padding positions (padding masked via **`-100`**). There is **no custom loss** and **no `label_smoothing`** in the current `Seq2SeqTrainingArguments`.

**`--weight-decay`** applies **AdamW weight decay** on parameters; that is **regularization**, not the sequence prediction loss.

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
| `--warmup-ratio` | `0.1` | Warmup as a fraction of total training steps. |
| `--logging-steps` | `100` | |
| `--eval-steps` | `500` | Eval every N steps (clamped when `--max-steps` set). |
| `--save-steps` | `500` | Save every N steps (clamped when `--max-steps` set). |
| `--save-total-limit` | `2` | Rolling checkpoint cap. |
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

## Example commands

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

- **`test`** split: used **once** after training for **`test_eval.json`** (teacher-forcing loss). Generation metrics (BLEU / ROUGE / BERTScore) still need a separate script or `compute_metrics` if you add them.
- **`--predict-with-generate`**: enables generation in eval but **does not** attach BLEU/ROUGE/BERTScore; add `compute_metrics` in code when those metrics are wired up.
- Transformers may emit warnings (for example deprecation of `warmup_ratio`, tied weights, or hub cache symlinks on Windows); they do not change the documented CLI behavior unless you upgrade libraries and APIs diverge.
