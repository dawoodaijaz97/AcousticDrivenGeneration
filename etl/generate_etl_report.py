from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table(df: pd.DataFrame) -> str:
    # Avoid external deps; generate a GitHub-flavored markdown table.
    def cell(v: Any) -> str:
        if isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=False, sort_keys=True)
        s = str(v)
        # Escape characters that break markdown tables.
        s = s.replace("|", "\\|").replace("\n", "<br>")
        return f"`{s}`" if any(ch in s for ch in ["{", "}", "[", "]", ":"]) else s

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ensure_figures(processed_dir: Path) -> None:
    """
    Make sure we have the per-split figures (features + lengths) under
    processed/<size>/dashboard_report.
    """
    dash_script = repo_root() / "etl" / "dashboard_1k" / "static_report.py"
    # The script is hard-coded to 1k, so for other sizes we generate figures here.
    out_dir = processed_dir / "dashboard_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # If figures already exist, skip.
    need = [
        out_dir / "train_features.png",
        out_dir / "train_lengths.png",
        out_dir / "val_features.png",
        out_dir / "val_lengths.png",
        out_dir / "test_features.png",
        out_dir / "test_lengths.png",
    ]
    if all(p.exists() for p in need):
        return

    # Generate plots using matplotlib if available.
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    feature_cols = [
        "breathing",
        "lips",
        "palate",
        "larynx",
        "monotonicity",
        "tongue",
        "intelligibility",
    ]

    def load_split(split: str) -> pd.DataFrame:
        p_parquet = processed_dir / f"{split}.parquet"
        p_csv = processed_dir / f"{split}.csv"
        if p_parquet.exists():
            return pd.read_parquet(p_parquet)
        return pd.read_csv(p_csv)

    for split in ["train", "val", "test"]:
        df = load_split(split)

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        for i, col in enumerate(feature_cols):
            ax = axes[i]
            ax.hist(df[col].dropna().values, bins=40)
            ax.set_title(col)
        for j in range(len(feature_cols), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{processed_dir.name} • {split}: feature histograms", y=0.98)
        fig.tight_layout()
        fig.savefig(out_dir / f"{split}_features.png", dpi=160)
        plt.close(fig)

        if "input_text" in df.columns and "target_text" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(df["input_text"].astype(str).str.len().values, bins=60)
            axes[0].set_title("input_text length (chars)")
            axes[1].hist(df["target_text"].astype(str).str.len().values, bins=60)
            axes[1].set_title("target_text length (chars)")
            fig.suptitle(f"{processed_dir.name} • {split}: text length histograms", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / f"{split}_lengths.png", dpi=160)
            plt.close(fig)


def main() -> None:
    root = repo_root()
    processed_root = root / "processed"
    report_dir = root / "etl" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "etl_report.md"

    sizes = ["1k", "10k", "100k"]
    summaries: Dict[str, Dict[str, Any]] = {}
    for s in sizes:
        summaries[s] = _read_json(processed_root / s / "summary.json")
        _ensure_figures(processed_root / s)

    # ---- Table 1: rows per split
    rows = []
    for s in sizes:
        for split in ["train", "val", "test"]:
            rows.append(
                {
                    "train_size": s,
                    "split": split,
                    "rows": _safe_get(summaries[s], "splits", split, "rows", default=0),
                    "is_real": _safe_get(summaries[s], "splits", split, "is_real_counts", default={}),
                    "group_counts": _safe_get(summaries[s], "splits", split, "group_counts", default={}),
                }
            )
    df_rows = pd.DataFrame(rows)

    # ---- Table 2: leakage drops
    df_leak = pd.DataFrame(
        [
            {
                "train_size": s,
                "val_dropped_due_to_train_overlap": _safe_get(
                    summaries[s], "leakage", "val_dropped_due_to_train_overlap", default=0
                ),
                "test_dropped_due_to_train_or_val_overlap": _safe_get(
                    summaries[s], "leakage", "test_dropped_due_to_train_or_val_overlap", default=0
                ),
            }
            for s in sizes
        ]
    )

    # ---- Table 3: length stats
    len_rows: List[Dict[str, Any]] = []
    for s in sizes:
        for split in ["train", "val", "test"]:
            il = _safe_get(summaries[s], "splits", split, "input_len_chars", default={})
            tl = _safe_get(summaries[s], "splits", split, "target_len_chars", default={})
            len_rows.append(
                {
                    "train_size": s,
                    "split": split,
                    "input_min": il.get("min"),
                    "input_p50": il.get("p50"),
                    "input_max": il.get("max"),
                    "target_min": tl.get("min"),
                    "target_p50": tl.get("p50"),
                    "target_max": tl.get("max"),
                }
            )
    df_lens = pd.DataFrame(len_rows)

    out_md = []
    out_md.append("# ETL report")
    out_md.append("")
    out_md.append("This report summarizes the outputs created by the ETL pipeline in `etl/`.")
    out_md.append("")
    out_md.append("## Dataset sizes and splits")
    out_md.append("")
    out_md.append(_md_table(df_rows[["train_size", "split", "rows", "is_real", "group_counts"]]))
    out_md.append("")
    out_md.append("## Leakage / overlap handling")
    out_md.append("")
    out_md.append(_md_table(df_leak))
    out_md.append("")
    out_md.append("## Text length stats (chars)")
    out_md.append("")
    out_md.append(_md_table(df_lens))
    out_md.append("")

    out_md.append("## Figures (per split)")
    out_md.append("")
    out_md.append("For each `train_size`, the following figures are generated under `processed/<train_size>/dashboard_report/`.")
    out_md.append("")
    for s in sizes:
        out_md.append(f"### {s}")
        out_md.append("")
        dash_abs = processed_root / s / "dashboard_report"
        # Make image paths relative to the report file location (etl/reports/...)
        dash_rel = Path(
            dash_abs.relative_to(root).as_posix()
        )  # e.g. processed/1k/dashboard_report
        img_base = Path("../..") / dash_rel  # from etl/reports → repo root
        out_md.append(f"- Features: `train_features.png`, `val_features.png`, `test_features.png`")
        out_md.append(f"- Lengths: `train_lengths.png`, `val_lengths.png`, `test_lengths.png`")
        out_md.append("")
        # Embed them (works in most markdown renderers)
        out_md.append(f"![{s} train features]({(img_base / 'train_features.png').as_posix()})")
        out_md.append(f"![{s} train lengths]({(img_base / 'train_lengths.png').as_posix()})")
        out_md.append("")
        out_md.append(f"![{s} val features]({(img_base / 'val_features.png').as_posix()})")
        out_md.append(f"![{s} val lengths]({(img_base / 'val_lengths.png').as_posix()})")
        out_md.append("")
        out_md.append(f"![{s} test features]({(img_base / 'test_features.png').as_posix()})")
        out_md.append(f"![{s} test lengths]({(img_base / 'test_lengths.png').as_posix()})")
        out_md.append("")

    report_path.write_text("\n".join(out_md) + "\n", encoding="utf-8")
    print(f"[OK] wrote {report_path}")


if __name__ == "__main__":
    main()

