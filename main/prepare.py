from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from main.io import (
    TrainSize,
    load_pipeline_splits,
    split_summaries,
    write_processed_splits,
)
from main.input_formats import resolve_input_format
from main.paths import repair_shell_collapsed_path, repo_root, resolve_under_repo
from main.prompts import resolve_prompt_style
from main.structured_targets import apply_structured_target_format


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare acoustic→report splits: load CSVs, normalize (via etl), serialize JSONL/Parquet.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (defaults to parent of the main package).",
    )
    p.add_argument(
        "--train-size",
        choices=("1k", "10k", "100k"),
        default="100k",
        help="Which simulated train_split subset to use.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for processed train/val/test parquet-or-csv + jsonl. Relative paths are under "
            "--repo-root (default: repository root), not the shell cwd. Prefer forward slashes "
            "(e.g. data/processed/1k): Git Bash strips backslashes in unquoted Windows paths."
        ),
    )
    p.add_argument(
        "--summaries-path",
        type=Path,
        default=None,
        help="Optional path to write split summaries as JSON (repo-relative if not absolute).",
    )
    p.add_argument(
        "--tokenize",
        action="store_true",
        help="Also build Hugging Face Arrow datasets with T5 tokenizer (requires transformers, datasets).",
    )
    p.add_argument(
        "--tokenizer-model",
        type=str,
        default="google/flan-t5-small",
        help="Model id for AutoTokenizer when --tokenize is set.",
    )
    p.add_argument(
        "--save-tokenized-dir",
        type=Path,
        default=None,
        help="Directory for DatasetDict.save_to_disk when --tokenize is set (repo-relative if not absolute).",
    )
    p.add_argument("--max-source-length", type=int, default=256)
    p.add_argument("--max-target-length", type=int, default=512)
    p.add_argument(
        "--input-format",
        choices=("compact-seven", "instructions-raw"),
        default="compact-seven",
        help=(
            "Encoder feature string: compact-seven = seven parsed category scores (default ETL input_text); "
            "instructions-raw = full CSV Instructions field (Speech_LLM / professor baseline)."
        ),
    )
    p.add_argument(
        "--prompt-style",
        choices=(
            "default",
            "flan-paper",
            "flan-paper-categories",
            "flan-paper-numeric-labels",
            "flan-paper-report-template",
        ),
        default="default",
        help=(
            "Encoder task prefix: default = mFDA biomarker instruction; "
            "flan-paper = 'Generate a report for:' (Phase 2 / B5); "
            "flan-paper-categories = paper prefix + explicit seven mFDA category hints (Phase 2 / B6); "
            "flan-paper-numeric-labels = paper prefix + deterministic key/value formatting "
            "for the seven categories (Phase 2 / B7); "
            "flan-paper-report-template = seven-slot Category (Severity): output template (Phase 2 / B19)."
        ),
    )
    p.add_argument(
        "--target-format",
        choices=("prose", "structured-seven-label"),
        default="prose",
        help=(
            "Decoder target format: prose = full clinical report (default); "
            "structured-seven-label = one 'Category: Severity' line per mFDA slot (B23)."
        ),
    )
    p.add_argument(
        "--lm-studio-base-url",
        type=str,
        default=None,
        help=(
            "Optional: save LM Studio server URL to lm_studio.json under --output-dir for eval workflows. "
            "Does not change tokenization (still uses --tokenizer-model from Hugging Face)."
        ),
    )
    p.add_argument(
        "--lm-studio-model",
        type=str,
        default=None,
        help="Optional: LM Studio model id (stored in lm_studio.json).",
    )
    return p.parse_args(argv)


def _log(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root()
    train_size: TrainSize = args.train_size

    out_repaired = repair_shell_collapsed_path(args.output_dir)
    if out_repaired != Path(args.output_dir):
        _log(
            f"expanded output-dir {args.output_dir!s} -> {out_repaired.as_posix()} "
            "(Git Bash removes \\ in unquoted paths; prefer data/processed/1k)"
        )
    out_dir = resolve_under_repo(args.output_dir, root)
    _log(f"repo_root        = {root}")
    _log(f"train_size       = {train_size}")
    _log(f"input_format     = {args.input_format}")
    _log(f"prompt_style     = {args.prompt_style}")
    _log(f"target_format    = {args.target_format}")
    input_format_key, input_column = resolve_input_format(args.input_format)
    task_prefix = resolve_prompt_style(args.prompt_style)
    _log(f"input_column     = {input_column!r}")
    _log(f"task_prefix      = {task_prefix!r}")
    _log(f"output_dir       = {out_dir}")

    _log("loading splits via etl ...")
    splits = load_pipeline_splits(root, train_size=train_size)
    if args.target_format == "structured-seven-label":
        _log("converting target_text -> structured seven-label format (reference_prose kept) ...")
        splits = apply_structured_target_format(splits)
    for name, df in splits.items():
        _log(f"  {name:<5} -> {len(df):>7,} rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    _log("writing parquet/jsonl ...")
    write_processed_splits(splits, out_dir)

    summaries = split_summaries(splits)
    summary_path = (
        resolve_under_repo(args.summaries_path, root)
        if args.summaries_path
        else (out_dir / "split_summaries.json").resolve()
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _log(f"wrote summaries  -> {summary_path}")

    prepare_config = {
        "train_size": train_size,
        "input_format": input_format_key,
        "input_column": input_column,
        "prompt_style": args.prompt_style,
        "target_format": args.target_format,
        "task_prefix": task_prefix,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "tokenizer_model": args.tokenizer_model if args.tokenize else None,
    }
    config_path = out_dir / "prepare_config.json"
    config_path.write_text(json.dumps(prepare_config, indent=2), encoding="utf-8")
    _log(f"wrote prepare_config -> {config_path}")

    if args.lm_studio_base_url:
        lm_payload = {
            "base_url": args.lm_studio_base_url.strip(),
            "model": args.lm_studio_model,
            "tokenizer_model": args.tokenizer_model if args.tokenize else None,
            "note": (
                "prepare still tokenizes with Hugging Face AutoTokenizer. "
                "For inference via LM Studio use: python -m main.eval_decode --backend lm-studio ..."
            ),
        }
        lm_path = out_dir / "lm_studio.json"
        lm_path.write_text(json.dumps(lm_payload, indent=2), encoding="utf-8")
        _log(f"wrote lm_studio -> {lm_path}")

    if args.tokenize:
        from transformers import AutoTokenizer

        from main.tokenization import dataframe_splits_to_dataset_dict, tokenize_dataset_dict

        tok_dir = (
            resolve_under_repo(args.save_tokenized_dir, root)
            if args.save_tokenized_dir
            else (out_dir / "tokenized").resolve()
        )
        tok_dir.mkdir(parents=True, exist_ok=True)
        _log(f"tokenizer_model  = {args.tokenizer_model}")
        _log(f"tokenized_dir    = {tok_dir}")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
        ddict = dataframe_splits_to_dataset_dict(
            splits,
            task_prefix=task_prefix,
            prompt_style=args.prompt_style,
            input_column=input_column,
        )
        tok_dict = tokenize_dataset_dict(
            ddict,
            tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        tok_dict.save_to_disk(str(tok_dir))
        _log("tokenization done.")

    _log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
