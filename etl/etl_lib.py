from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd


SplitName = Literal["train", "val", "test"]


FEATURE_ORDER: Tuple[str, ...] = (
    "Breathing",
    "Lips",
    "Palate",
    "Larynx",
    "Monotonicity",
    "Tongue",
    "Intelligibility",
)

FEATURE_COLS: Tuple[str, ...] = (
    "breathing",
    "lips",
    "palate",
    "larynx",
    "monotonicity",
    "tongue",
    "intelligibility",
)

_FEATURE_TO_COL = dict(zip(FEATURE_ORDER, FEATURE_COLS))


def _compile_feature_regex() -> re.Pattern[str]:
    # Handles both:
    # - "Breathing: 78. Lips: 1.75. ..."
    # - "Breathing: 53 Lips: 0.89 Palate: 0.67 ..."
    # Allows optional punctuation after value.
    names = "|".join(map(re.escape, FEATURE_ORDER))
    return re.compile(
        rf"(?P<name>{names})\s*:\s*(?P<value>[-+]?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )


_FEATURE_RE = _compile_feature_regex()


def parse_instructions_features(instructions: str) -> Dict[str, float]:
    found: Dict[str, float] = {}
    for m in _FEATURE_RE.finditer(instructions):
        name_raw = m.group("name")
        value = float(m.group("value"))
        name_norm = next((n for n in FEATURE_ORDER if n.lower() == name_raw.lower()), name_raw)
        found[_FEATURE_TO_COL[name_norm]] = value
    return found


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_input_text(row: pd.Series) -> str:
    # Fixed order, fixed formatting for stability across reruns.
    parts: List[str] = []
    for feat_name in FEATURE_ORDER:
        col = _FEATURE_TO_COL[feat_name]
        val = row[col]
        parts.append(f"{feat_name}: {val:.6g}")
    return " ".join(parts)


def make_example_hash(input_text: str, target_text: str) -> str:
    blob = (input_text + "\n\n" + target_text).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class LoadSpec:
    csv_path: Path
    split: SplitName
    is_real: bool


def load_split_csv(spec: LoadSpec, repo_root: Path) -> pd.DataFrame:
    df = pd.read_csv(spec.csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalize schemas:
    if "Group" not in df.columns:
        df["Group"] = None
    df = df.rename(columns={"ID": "sample_id", "Group": "group", "Instructions": "instructions_raw", "Report": "report_raw"})

    keep = ["sample_id", "group", "instructions_raw", "report_raw"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {spec.csv_path}")

    df = df[keep].copy()
    df["split"] = spec.split
    df["is_real"] = bool(spec.is_real)

    # Track provenance
    df["source_file"] = str(spec.csv_path.relative_to(repo_root).as_posix())
    df["source_row"] = range(len(df))

    # Clean text fields (minimal)
    df["instructions_raw"] = df["instructions_raw"].astype(str).map(normalize_whitespace)
    df["report_raw"] = df["report_raw"].astype(str).map(normalize_whitespace)

    # Parse features
    feat_rows = df["instructions_raw"].map(parse_instructions_features)
    for col in FEATURE_COLS:
        df[col] = feat_rows.map(lambda d: d.get(col, float("nan")))

    # Validate parsed features
    parsed_ok = df[list(FEATURE_COLS)].notna().all(axis=1)
    df = df.loc[parsed_ok].copy()

    # Build model fields
    df["input_text"] = df.apply(make_input_text, axis=1)
    df["target_text"] = df["report_raw"]
    df["example_hash"] = [make_example_hash(i, t) for i, t in zip(df["input_text"], df["target_text"], strict=True)]

    # Deterministic deduplication (keep first by source_file/source_row)
    df = df.sort_values(["source_file", "source_row"]).drop_duplicates(subset=["example_hash"], keep="first")

    # Normalize group
    df["group"] = df["group"].where(df["group"].notna(), None)
    df.loc[df["group"].astype(str).str.lower().isin(["none", "nan"]), "group"] = None

    return df.reset_index(drop=True)


def write_outputs(
    *,
    df: pd.DataFrame,
    out_dir: Path,
    split: SplitName,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{split}.parquet"
    csv_path = out_dir / f"{split}.csv"
    jsonl_path = out_dir / f"{split}.jsonl"

    # Prefer Parquet when available; fall back to CSV when pyarrow/fastparquet
    # is not installed in the current environment.
    try:
        df.to_parquet(parquet_path, index=False)
    except ImportError:
        df.to_csv(csv_path, index=False, encoding="utf-8")

    # JSONL: only the fields that matter for training + tracing.
    jsonl_cols = [
        "example_hash",
        "sample_id",
        "split",
        "is_real",
        "group",
        *FEATURE_COLS,
        "input_text",
        "target_text",
        "source_file",
        "source_row",
    ]
    records = df[jsonl_cols].to_dict(orient="records")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["rows"] = int(len(df))
    out["is_real_counts"] = {str(k): int(v) for k, v in df["is_real"].value_counts(dropna=False).items()}
    out["group_counts"] = {str(k): int(v) for k, v in df["group"].fillna("None").value_counts(dropna=False).items()}
    out["feature_missing_rows"] = int(df[list(FEATURE_COLS)].isna().any(axis=1).sum())
    out["input_len_chars"] = {
        "min": int(df["input_text"].str.len().min()),
        "p50": int(df["input_text"].str.len().median()),
        "max": int(df["input_text"].str.len().max()),
    }
    out["target_len_chars"] = {
        "min": int(df["target_text"].str.len().min()),
        "p50": int(df["target_text"].str.len().median()),
        "max": int(df["target_text"].str.len().max()),
    }
    return out

