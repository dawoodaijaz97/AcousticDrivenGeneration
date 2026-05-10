# ETL pipeline

This folder contains a small ETL pipeline that:

- Loads the project CSV splits from `data/Data_splits/`
- Normalizes simulated vs real schemas into one canonical schema
- Parses the 7 acoustic features from the `Instructions` string into numeric columns
- Performs basic validation + deterministic deduplication
- Writes `processed/` datasets as **Parquet** (and **JSONL**) + a `summary.json`

## Quick start

Run ETL for all train sizes (1k/10k/100k):

```bash
python etl/run_etl.py
```

Run ETL for a single train size:

```bash
python etl/run_etl.py --train-size 10k
```

Outputs are written to:

- `processed/<train_size>/train.parquet`
- `processed/<train_size>/val.parquet`
- `processed/<train_size>/test.parquet`
- `processed/<train_size>/train.jsonl`
- `processed/<train_size>/val.jsonl`
- `processed/<train_size>/test.jsonl`
- `processed/<train_size>/summary.json`

## Canonical schema (output columns)

- `sample_id`: original `ID`
- `split`: `train|val|test`
- `is_real`: boolean
- `group`: `PD|HC|None`
- `instructions_raw`: original `Instructions`
- `report_raw`: original `Report`
- `breathing`, `lips`, `palate`, `larynx`, `monotonicity`, `tongue`, `intelligibility`: parsed floats
- `input_text`: normalized prompt string (fixed feature order)
- `target_text`: report string (same as `report_raw`, lightly whitespace-normalized)
- `source_file`: original CSV path (relative)
- `source_row`: 0-based row index within the CSV (excluding header)
- `example_hash`: stable SHA256 of `(input_text + \\n\\n + target_text)`

## Reports

- **Markdown** (tables + figure links): run `python etl/generate_etl_report.py` → `etl/reports/etl_report.md`
- **Figures** (PNG): run `python etl/make_processed_figures.py --train-size all` → `processed/<size>/dashboard_report/*.png`
- **Single PDF** (tables + all dashboard figures): after ETL and figures exist, run:

```bash
conda run -n AcousticDrivenGeneration python etl/generate_etl_pdf.py
```

Default output: `etl/reports/etl_report.pdf`. Custom path: `python etl/generate_etl_pdf.py -o path/to/report.pdf`

