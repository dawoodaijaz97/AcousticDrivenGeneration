from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import pandas as pd

# Ensure sibling package ``etl`` is importable when running as ``python -m main.prepare``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from etl.etl_lib import LoadSpec, load_split_csv, summarize, write_outputs

from main.paths import data_splits_dir

TrainSize = Literal["1k", "10k", "100k"]
SplitName = Literal["train", "val", "test"]


def train_csv_path(train_size: TrainSize, root: Path | None = None) -> Path:
    base = data_splits_dir(root)
    return base / f"{train_size}_samples_balanced" / "train_split.csv"


def val_csv_path(root: Path | None = None) -> Path:
    return data_splits_dir(root) / "val_split.csv"


def test_csv_path(root: Path | None = None) -> Path:
    return data_splits_dir(root) / "test_split.csv"


def load_pipeline_splits(
    root: Path | None = None,
    *,
    train_size: TrainSize = "100k",
) -> dict[SplitName, pd.DataFrame]:
    """Load train / val (synthetic) and test (real reports) tables via ETL helpers."""
    r = root if root is not None else _ROOT
    specs: tuple[LoadSpec, ...] = (
        LoadSpec(train_csv_path(train_size, r), "train", False),
        LoadSpec(val_csv_path(r), "val", False),
        LoadSpec(test_csv_path(r), "test", True),
    )
    out: dict[SplitName, pd.DataFrame] = {}
    for spec in specs:
        out[spec.split] = load_split_csv(spec, r)
    return out


def write_processed_splits(
    splits: dict[SplitName, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write parquet (if available), CSV fallback, and JSONL per split."""
    for name, df in splits.items():
        write_outputs(df=df, out_dir=output_dir, split=name)


def split_summaries(splits: dict[SplitName, pd.DataFrame]) -> dict[str, dict]:
    return {k: summarize(v) for k, v in splits.items()}
