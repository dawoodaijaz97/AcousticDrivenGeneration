from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from etl.etl_lib import LoadSpec, load_split_csv, summarize, write_outputs  # noqa: E402


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def get_specs(repo_root: Path, train_size: str) -> List[LoadSpec]:
    data_root = repo_root / "data" / "Data_splits"
    train_path = data_root / f"{train_size}_samples_balanced" / "train_split.csv"
    val_path = data_root / "val_split.csv"
    test_path = data_root / "test_split.csv"

    return [
        LoadSpec(csv_path=train_path, split="train", is_real=False),
        LoadSpec(csv_path=val_path, split="val", is_real=False),
        LoadSpec(csv_path=test_path, split="test", is_real=True),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-size",
        default="all",
        choices=["all", "1k", "10k", "100k"],
        help="Which simulated train split to use (folder suffix).",
    )
    args = ap.parse_args()

    repo_root = repo_root_from_this_file()
    out_root = repo_root / "processed"

    train_sizes = ["1k", "10k", "100k"] if args.train_size == "all" else [args.train_size]

    for train_size in train_sizes:
        specs = get_specs(repo_root, train_size=train_size)
        frames: Dict[str, pd.DataFrame] = {}
        for spec in specs:
            frames[spec.split] = load_split_csv(spec, repo_root=repo_root)

        # Leakage guard:
        # Simulated splits reuse `ID` values, so we cannot rely on `sample_id` uniqueness.
        # Instead, we prevent exact content leakage by removing examples whose `example_hash`
        # overlaps across splits.
        train_hashes = set(frames["train"]["example_hash"])
        val_hashes = set(frames["val"]["example_hash"])
        test_hashes = set(frames["test"]["example_hash"])

        val_overlap = val_hashes.intersection(train_hashes)
        test_overlap = test_hashes.intersection(train_hashes.union(val_hashes))
        if val_overlap:
            frames["val"] = frames["val"].loc[~frames["val"]["example_hash"].isin(val_overlap)].reset_index(drop=True)
        if test_overlap:
            frames["test"] = frames["test"].loc[~frames["test"]["example_hash"].isin(test_overlap)].reset_index(drop=True)

        out_dir = out_root / train_size
        for split, df in frames.items():
            write_outputs(df=df, out_dir=out_dir, split=split)  # type: ignore[arg-type]

        summary = {
            "train_size": train_size,
            "paths": {
                k: {
                    "parquet": str((out_dir / f"{k}.parquet").relative_to(repo_root).as_posix()),
                    "csv_fallback": str((out_dir / f"{k}.csv").relative_to(repo_root).as_posix()),
                    "jsonl": str((out_dir / f"{k}.jsonl").relative_to(repo_root).as_posix()),
                }
                for k in frames.keys()
            },
            "splits": {k: summarize(v) for k, v in frames.items()},
            "leakage": {
                "val_dropped_due_to_train_overlap": len(val_overlap),
                "test_dropped_due_to_train_or_val_overlap": len(test_overlap),
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(f"[OK] wrote processed datasets for train_size={train_size} to {out_dir}")


if __name__ == "__main__":
    main()

