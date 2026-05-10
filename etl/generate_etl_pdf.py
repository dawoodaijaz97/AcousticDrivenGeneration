"""
Build a single PDF summarizing ETL outputs: tables from processed/*/summary.json
and figures from processed/*/dashboard_report/*.png.

Run after ETL + figure generation, or this script will call _ensure_figures for each size.

Usage (from repo root):
  conda run -n AcousticDrivenGeneration python etl/generate_etl_pdf.py
  conda run -n AcousticDrivenGeneration python etl/generate_etl_pdf.py -o etl/reports/my_etl.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from etl.generate_etl_report import (  # noqa: E402
    _ensure_figures,
    _read_json,
    _safe_get,
    repo_root,
)


def _df_splits_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("is_real", "group_counts"):
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True) if isinstance(x, dict) else str(x)
            )
    return out


def _add_table_page(pdf, title: str, df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)

    cells: List[List[str]] = []
    for _, row in df.iterrows():
        cells.append([str(v) for v in row.tolist()])
    col_labels = [str(c) for c in df.columns]

    tbl = ax.table(
        cellText=cells,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.05, 1.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_title_page(pdf, lines: List[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    y = 0.92
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        size = 18 if i == 0 else 11
        ax.text(0.08, y, line, fontsize=size, fontweight=weight, transform=ax.transAxes, va="top")
        y -= 0.06 if i == 0 else 0.045
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_text_pages(
    pdf,
    title: Optional[str],
    paragraphs: List[str],
    *,
    line_chars: int = 92,
    fontsize: int = 10,
    title_fontsize: int = 14,
) -> None:
    """One or more letter pages of wrapped body text (continues across pages if needed)."""
    import matplotlib.pyplot as plt

    line_h = 0.028
    para_gap = 0.018
    top = 0.94
    bottom = 0.07
    x0 = 0.08
    show_title = bool(title)

    def new_fig(*, with_section_title: bool) -> tuple[Any, Any, float]:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        y = top
        if with_section_title and title:
            ax.text(
                x0,
                y,
                title,
                fontsize=title_fontsize,
                fontweight="bold",
                transform=ax.transAxes,
                va="top",
            )
            y -= 0.055
        return fig, ax, y

    fig, ax, y = new_fig(with_section_title=show_title)
    if show_title and not paragraphs:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    for para in paragraphs:
        if not para.strip():
            y -= para_gap
            continue
        wrapped = textwrap.wrap(para.strip(), width=line_chars)
        for line in wrapped:
            if y < bottom:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax, y = new_fig(with_section_title=False)
            ax.text(x0, y, line, fontsize=fontsize, transform=ax.transAxes, va="top")
            y -= line_h
        y -= para_gap

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_image_page(pdf, title: str, image_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    if not image_path.is_file():
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_dataframes(
    summaries: Dict[str, Dict[str, Any]], sizes: List[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    df_leak = pd.DataFrame(
        [
            {
                "train_size": s,
                "val_dropped": _safe_get(
                    summaries[s], "leakage", "val_dropped_due_to_train_overlap", default=0
                ),
                "test_dropped": _safe_get(
                    summaries[s], "leakage", "test_dropped_due_to_train_or_val_overlap", default=0
                ),
            }
            for s in sizes
        ]
    )

    len_rows: List[Dict[str, Any]] = []
    for s in sizes:
        for split in ["train", "val", "test"]:
            il = _safe_get(summaries[s], "splits", split, "input_len_chars", default={})
            tl = _safe_get(summaries[s], "splits", split, "target_len_chars", default={})
            len_rows.append(
                {
                    "train_size": s,
                    "split": split,
                    "in_min": il.get("min"),
                    "in_p50": il.get("p50"),
                    "in_max": il.get("max"),
                    "tgt_min": tl.get("min"),
                    "tgt_p50": tl.get("p50"),
                    "tgt_max": tl.get("max"),
                }
            )
    df_lens = pd.DataFrame(len_rows)
    return df_rows, df_leak, df_lens


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate ETL summary PDF.")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: etl/reports/etl_report.pdf)",
    )
    args = ap.parse_args()

    root = repo_root()
    processed_root = root / "processed"
    report_dir = root / "etl" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = args.output if args.output is not None else report_dir / "etl_report.pdf"
    out_pdf = out_pdf.resolve()

    default_sizes = ["1k", "10k", "100k"]
    found = sorted({p.parent.name for p in processed_root.glob("*/summary.json")})
    sizes = [s for s in default_sizes if s in found]
    if not sizes:
        sizes = sorted(found)
    if not sizes:
        raise FileNotFoundError(
            f"No summary.json under {processed_root}. Run ETL first: python etl/run_etl.py"
        )

    summaries: Dict[str, Dict[str, Any]] = {}
    for s in sizes:
        summary_path = processed_root / s / "summary.json"
        summaries[s] = _read_json(summary_path)
        _ensure_figures(processed_root / s)

    df_rows, df_leak, df_lens = _build_dataframes(summaries, sizes)
    df_rows_disp = _df_splits_display(df_rows)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf) as pdf:
        try:
            pdf.metadata["Title"] = "AcousticDrivenGeneration — ETL report"
            pdf.metadata["Subject"] = "Processed splits, leakage handling, length stats, dashboard figures"
        except Exception:
            pass

        _add_title_page(
            pdf,
            [
                "AcousticDrivenGeneration",
                "ETL report (processed data)",
                "",
                f"Source: {processed_root.as_posix()}",
                "Tables: split sizes, simulated vs real counts, group counts (test),",
                "         overlap drops (example_hash), text length stats.",
                "Figures: feature histograms and input/target length histograms per split.",
                "",
                f"Generated: {pd.Timestamp.now(tz=None).strftime('%Y-%m-%d %H:%M')}",
            ],
        )

        _add_text_pages(
            pdf,
            "What this report contains",
            [
                "This document summarizes the output of the ETL pipeline: CSV splits under "
                "data/Data_splits/ are normalized, acoustic features are parsed from the "
                "Instructions field, and artifacts are written under processed/<train_size>/ "
                "(CSV/JSONL plus summary.json).",
                "The tables quantify how many rows exist in each split, how many examples were "
                "removed to avoid exact duplicate leakage across splits, and how long the "
                "model input and target texts are. The figures show distributions of the seven "
                "parsed features and of character lengths for train, validation, and test.",
                f"This PDF includes the following simulated train sizes found on disk: {', '.join(sizes)}.",
            ],
        )

        _add_text_pages(
            pdf,
            "Table 1 — Dataset sizes and splits",
            [
                "Each row is one combination of train_size (which simulated training subset was "
                "used when building that processed folder) and split name.",
                "The rows column is the number of examples kept after parsing all seven acoustic "
                "features and dropping rows with missing values. Simulated train/val rows have "
                "is_real False; the test split is real clinical data with is_real True.",
                "group_counts is only meaningful for the real test split: PD vs HC counts. "
                "Simulated splits have no group label (shown as None).",
            ],
        )
        _add_table_page(pdf, "Dataset sizes and splits", df_rows_disp)

        _add_text_pages(
            pdf,
            "Table 2 — Leakage / overlap handling",
            [
                "Simulated train and val CSVs reuse numeric IDs, so overlap by sample_id is not "
                "a reliable leakage check. The ETL pipeline deduplicates within each file and "
                "then removes val (and test) rows whose example_hash matches an example already "
                "present in train (or val, for test).",
                "example_hash is SHA-256 of the normalized input_text plus the target report text, "
                "so it flags exact duplicate prompt+report pairs across splits.",
                "val_dropped is the count of validation rows removed for matching train; "
                "test_dropped is the count of real test rows removed for matching train or val "
                "(usually zero). Small non-zero val_dropped values are expected when the same "
                "synthetic example appears in both train and val.",
            ],
        )
        _add_table_page(
            pdf,
            "Leakage / overlap handling (rows removed from val/test if same example_hash as train)",
            df_leak,
        )

        _add_text_pages(
            pdf,
            "Table 3 — Text length statistics",
            [
                "input_text is the fixed-order string built from the seven parsed features "
                "(Breathing, Lips, Palate, Larynx, Monotonicity, Tongue, Intelligibility). "
                "target_text is the clinical report string.",
                "Columns in_min / in_p50 / in_max are the minimum, median, and maximum length "
                "of input_text in characters for that split. tgt_* are the same for target_text.",
                "Use this table to sanity-check tokenizer limits and to compare simulated vs real "
                "test report lengths.",
            ],
        )
        _add_table_page(pdf, "Text length stats (characters)", df_lens)

        _add_text_pages(
            pdf,
            "Figures — How to read the histograms",
            [
                "For each processed train size (1k, 10k, 100k when present), six images are "
                "included: train, val, and test, each with a features grid and a lengths pair.",
                "Features plots: one histogram per parsed feature (breathing through intelligibility). "
                "They show the empirical distribution of values in that split after ETL.",
                "Lengths plots: left histogram is input_text length in characters; right is "
                "target_text length. Together they show how compact prompts are relative to reports.",
                "Simulated train/val distributions may differ slightly between train_size folders "
                "because the underlying sampled rows differ; the real test images are the same "
                "across folders (same 96 real examples), repeated for each train_size section.",
            ],
        )

        figure_order = [
            ("train", "features"),
            ("train", "lengths"),
            ("val", "features"),
            ("val", "lengths"),
            ("test", "features"),
            ("test", "lengths"),
        ]
        for train_size in sizes:
            _add_text_pages(
                pdf,
                f"Figures — train_size = {train_size}",
                [
                    f"The following pages show ETL dashboard plots for the processed/{train_size} "
                    "folder: synthetic train and val (unless noted) and real test.",
                    "Order: train features, train lengths, val features, val lengths, test features, "
                    "test lengths. If a PNG is missing, that page is skipped.",
                ],
            )
            dash = processed_root / train_size / "dashboard_report"
            for split, kind in figure_order:
                fname = f"{split}_{kind}.png"
                path = dash / fname
                title = f"{train_size} — {split} — {kind}"
                _add_image_page(pdf, title, path)

    print(f"[OK] wrote {out_pdf}")


if __name__ == "__main__":
    main()
