# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Fine-tunes T5 / Flan-T5 seq2seq models to generate mFDA-style clinical speech
reports from seven numeric acoustic biomarkers (Breathing, Lips, Palate,
Larynx, Monotonicity, Tongue, Intelligibility). Training data is
synthetic/simulated prompt-report pairs; the held-out test split is real
PD/HC clinical reports. This reproduces/extends a paper pipeline (see
`docs/paper.pdf`, `docs/paper-overview.md`).

**`docs/` and `data/` are gitignored (not tracked in git) but present and
authoritative on disk** — always read them directly rather than assuming
they're absent because `git ls-files` won't show them.

## Commands

Environment (conda env name `AcousticDrivenGeneration`; see `requirements.txt`
— PyTorch is intentionally *not* pinned there, install it separately to match
your CUDA/CPU build):

```powershell
conda activate AcousticDrivenGeneration
python -c "import sys; print(sys.executable)"   # sanity check it's the env's python
```

Run tests (from repo root; `main/` is imported directly, no src layout, no
pytest config file):

```powershell
python -m pytest test/
python -m pytest test/test_structured_report.py::StructuredReportTests::test_force_template_fills_missing_slots  # single test
```

Note: `test/test.py` is not a real pytest test — it's a standalone CUDA smoke
script (`python test/test.py`) with a bare `print` at module scope; pytest
will error collecting it as a test class/function, so it's normally run
directly instead.

Core pipeline, run in order:

```powershell
# 1. ETL: raw CSVs -> canonical processed/<size>/ (optional, main.prepare can read raw CSVs directly)
python etl\run_etl.py --train-size 1k        # or 10k / 100k / all (default)

# 2. Prepare: canonical splits -> tokenized HF DatasetDict
python -m main.prepare --output-dir data\processed\flan-t5-base\100k --train-size 100k `
  --tokenize --tokenizer-model google/flan-t5-base

# 3. Train (Seq2SeqTrainer; add --require-gpu on a real GPU box)
python -m main.train --tokenized-dir data\processed\flan-t5-base\100k\tokenized `
  --output-dir runs\flan-t5-base\100k --model-name google/flan-t5-base `
  --num-train-epochs 5 --learning-rate 5e-4 --weight-decay 0.01

# 4. Decode eval (ROUGE/BLEU/BERTScore vs paper Table 4) — GPU only, CPU decode times out
python -m main.eval_decode --tokenized-dir data\processed\flan-t5-base\100k\tokenized `
  --model-path runs\flan-t5-base\100k\final_model --tokenizer-model google/flan-t5-base `
  --output-json runs\flan-t5-base\100k\test_decode_metrics.json
```

Full flag references and worked examples: [docs/data-pipeline.md](docs/data-pipeline.md),
[docs/train.md](docs/train.md), [docs/eval-decode.md](docs/eval-decode.md), [etl/README.md](etl/README.md).

HPC (NHR@FAU TinyGPU) SSH/Slurm/conda/HF-cache setup: [docs/hpc-commands.md](docs/hpc-commands.md)
and `.cursor/rules/hpc-nhr-fau.mdc` (always-apply cursor rule with the exact
account, paths, and gotchas — e.g. no `--fp16` on V100, `.tinygpu`-suffixed
Slurm commands, 24h wall-time cap).

## Architecture

Three sequential stages, each reusable independently, that share one parsing
core (`etl/etl_lib.py`) so nothing re-parses raw strings downstream:

```
data/Data_splits/*.csv  →  etl/  →  main/(prepare)  →  main/(train)  →  main/(eval_decode / eval_structured)
   (raw, simulated +        canonical         tokenized DatasetDict      fine-tuned            metrics JSON
    real test reports)      parquet/jsonl      (source_text/labels)      checkpoint
```

- **`etl/etl_lib.py`** — the single source of truth for schema normalization,
  regex-parsing the 7 biomarkers out of the raw `Instructions` string,
  building canonical `input_text`/`target_text`, hashing (`example_hash =
  sha256(input+target)`) and dedup. `etl/run_etl.py` batches this across
  `{1k,10k,100k}` and adds a cross-split leakage guard. `main/io.py` calls the
  *same* `load_split_csv` so `main.prepare` can bypass `etl/run_etl.py`
  entirely and read raw CSVs straight through.
- **`main/prompts.py`** — all encoder task-prefix variants live here as
  `PROMPT_STYLES` (`default`, `flan-paper`, `flan-paper-categories`,
  `flan-paper-numeric-labels`, `flan-paper-report-template`). `FEATURE_ORDER`
  (the 7 categories, fixed order) is defined once here and imported by
  `report_severity.py`, `structured_report.py`, `structured_targets.py` — the
  canonical order for parsing and for template generation.
- **`main/tokenization.py`** — builds the HF `DatasetDict`, tokenizes
  `source_text`/`target_text`, masks label padding to `-100`.
- **`main/train.py`** — thin wrapper around `Seq2SeqTrainer`. Supports PEFT
  LoRA (`--lora-rank`, merged into `final_model/` on save) and
  `--freeze-encoder` (mutually exclusive with LoRA), plus severity-based
  oversampling (`--oversample-group`, `--oversample-severity-min`, uses
  `main/report_severity.py` to find rows meeting a severity threshold).
- **`main/eval_decode.py`** — the free-form decode path: beam search →
  ROUGE/BLEU/BERTScore vs paper Table 4, optional PD/HC group breakdown.
  **Model selection must use decode `AVG` from this script's output, not
  `test_loss`** — training loss and generation quality have diverged
  repeatedly in this project's experiment log (see below).
- **`main/structured_report.py` / `main/structured_targets.py` /
  `main/report_severity.py` / `main/eval_structured.py`** — the newer B23
  "seven-slot" structured path: force free text into a mandatory
  `Category (Severity): description` skeleton per category, or convert to/from
  a compact `Category: Severity` label-only target, then score label accuracy
  separately from prose-decode quality. `main/eval_structured.py` is the
  decode-eval entry point for models trained on these structured targets
  (parallel to, and built on top of, `main/eval_decode.py`).
- **`main/analyze_pd_decode.py` / `main/audit_report_distribution.py`** —
  post-hoc analysis over already-generated decode JSON: PD-vs-HC category/
  severity gaps, and synthetic-vs-real target-text distribution audits.
- **`main/lm_studio.py`** — optional helper for hitting a local LM Studio
  OpenAI-compatible server; unrelated to the T5 training/eval path.

### Experiment naming and provenance

Runs are labeled with IDs (`B5`, `B11`, `B14`... for flan-t5-base; `S`/`L`/`N`
prefixes for small/large/non-Flan) tracked across three docs that must stay in
sync when you add an experiment: `docs/train.md` (recipe + current-best
table), `docs/training_progress.md` (results log), and `docs/Model
Improvement Plan.md` (checklist/roadmap). Slurm scripts under `scripts/hpc/`
are named after the same IDs (e.g. `train_flan_t5_base_b19_report_template_lora32_a100.slurm`).
Before proposing a new training config, check `docs/train.md`'s "Closed
levers" table — several axes (label smoothing, weight-decay off-0.01,
several prompt variants) were already tried and regressed; don't re-run them
without a new hypothesis.

### `runs/` is mostly gitignored on purpose

Checkpoints and `final_model/` stay local/HPC-only. Only
`test_decode_metrics.json`, `test_eval.json`, `pd_analysis.json`, and `*.png`
under `runs/**` are tracked in git (see `.gitignore`), so that decode-metric
history survives across machines without checking in multi-GB model weights.
When pulling metrics off HPC, `git add -f` is required the first time for a
given run directory (glob won't expand automatically against gitignored
trees) — see [docs/hpc-commands.md](docs/hpc-commands.md).
