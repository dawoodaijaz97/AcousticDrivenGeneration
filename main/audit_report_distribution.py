"""D1 — synthetic-vs-real mFDA report distribution audit.

Compares train/val synthetic ``target_text`` against real test references:
category slot coverage, severity distributions, length, duplication, and PD/HC
wording on the real test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from etl.etl_lib import LoadSpec, load_split_csv
from main.paths import repo_root, resolve_under_repo
from main.prompts import FEATURE_ORDER
from main.report_severity import SEVERITIES, parse_severities
from main.structured_report import aggregate_structure_metrics, structure_metrics_for_text


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit synthetic vs real mFDA report target distributions (D1).",
    )
    p.add_argument("--train", type=Path, required=True, help="Synthetic train_split.csv")
    p.add_argument("--val", type=Path, required=True, help="Synthetic val_split.csv")
    p.add_argument("--test", type=Path, required=True, help="Real test_split.csv")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write audit JSON (default: analysis/synthetic_real_report_audit.json).",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Write Markdown summary (default: analysis/synthetic_real_report_audit.md).",
    )
    p.add_argument(
        "--sample-duplicate-examples",
        type=int,
        default=5,
        help="Max repeated target_text examples to include per split.",
    )
    return p.parse_args(argv)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _severity_histogram(texts: list[str]) -> dict[str, dict[str, float]]:
    """Per-category severity rates among rows where the slot is present."""
    counts: dict[str, Counter[str]] = {cat: Counter() for cat in FEATURE_ORDER}
    for text in texts:
        sevs = parse_severities(text)
        for cat in FEATURE_ORDER:
            sev = sevs.get(cat)
            if sev in SEVERITIES:
                counts[cat][sev] += 1
    out: dict[str, dict[str, float]] = {}
    for cat in FEATURE_ORDER:
        total = sum(counts[cat].values())
        if total == 0:
            out[cat] = {sev: 0.0 for sev in SEVERITIES}
            continue
        out[cat] = {sev: round(counts[cat][sev] / total, 4) for sev in SEVERITIES}
    return out


def _duplicate_stats(texts: list[str], *, max_examples: int) -> dict[str, Any]:
    counts = Counter(texts)
    repeated = [(text, n) for text, n in counts.items() if n > 1]
    repeated.sort(key=lambda x: (-x[1], -len(x[0])))
    unique = len(counts)
    return {
        "n": len(texts),
        "unique_targets": unique,
        "duplicate_rate": round(1.0 - unique / len(texts), 4) if texts else 0.0,
        "max_duplicate_count": max(counts.values()) if counts else 0,
        "top_duplicates": [
            {"count": n, "preview": text[:240] + ("…" if len(text) > 240 else "")}
            for text, n in repeated[:max_examples]
        ],
    }


def _category_presence_rates(texts: list[str]) -> dict[str, float]:
    n = len(texts) or 1
    tallies = {cat: 0 for cat in FEATURE_ORDER}
    for text in texts:
        sevs = parse_severities(text)
        for cat in FEATURE_ORDER:
            if sevs.get(cat) is not None:
                tallies[cat] += 1
    return {cat: round(tallies[cat] / n, 4) for cat in FEATURE_ORDER}


def audit_dataframe(df: pd.DataFrame, *, max_duplicate_examples: int) -> dict[str, Any]:
    texts = [str(t) for t in df["target_text"].astype(str)]
    groups = [None if pd.isna(g) else str(g) for g in df.get("group", pd.Series([None] * len(df)))]
    structure = aggregate_structure_metrics(texts, groups if any(g is not None for g in groups) else None)

    per_example = [structure_metrics_for_text(t) for t in texts]
    all_seven = sum(1 for m in per_example if m["all_7_slots"]) / len(per_example) if per_example else 0.0

    lengths = [len(t) for t in texts]
    report: dict[str, Any] = {
        "n": len(texts),
        "is_real": bool(df["is_real"].iloc[0]) if len(df) else False,
        "group_counts": {
            str(k): int(v)
            for k, v in df["group"].fillna("None").value_counts(dropna=False).items()
        },
        "target_len_chars": {
            "min": min(lengths) if lengths else 0,
            "p50": int(pd.Series(lengths).median()) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(_mean([float(x) for x in lengths]), 1) if lengths else 0.0,
        },
        "structure_metrics": structure,
        "mean_category_coverage": structure.get("category_coverage", 0.0),
        "all_7_slots_rate": round(all_seven, 4),
        "category_presence_rate": _category_presence_rates(texts),
        "severity_distribution": _severity_histogram(texts),
        "duplicates": _duplicate_stats(texts, max_examples=max_duplicate_examples),
    }

    if any(g in ("PD", "HC") for g in groups):
        by_group: dict[str, Any] = {}
        for label in ("PD", "HC"):
            idxs = [i for i, g in enumerate(groups) if g == label]
            if not idxs:
                continue
            subset_texts = [texts[i] for i in idxs]
            by_group[label] = {
                "n": len(subset_texts),
                "mean_category_coverage": round(
                    _mean([structure_metrics_for_text(t)["category_coverage"] for t in subset_texts]), 4
                ),
                "category_presence_rate": _category_presence_rates(subset_texts),
                "severity_distribution": _severity_histogram(subset_texts),
                "target_len_chars": {
                    "p50": int(pd.Series([len(t) for t in subset_texts]).median()),
                    "mean": round(_mean([float(len(t)) for t in subset_texts]), 1),
                },
            }
        report["by_group"] = by_group

    return report


def _load_split_csv(path: Path, split: Literal["train", "val", "test"], is_real: bool) -> pd.DataFrame:
    root = repo_root()
    spec = LoadSpec(resolve_under_repo(path, root), split, is_real)
    return load_split_csv(spec, root)


def _compare_splits(synthetic: dict[str, Any], real: dict[str, Any]) -> dict[str, Any]:
    syn_cov = float(synthetic["mean_category_coverage"])
    real_cov = float(real["mean_category_coverage"])
    syn_presence = synthetic["category_presence_rate"]
    real_presence = real["category_presence_rate"]
    breathing_gap = round(float(syn_presence.get("Breathing", 0.0)) - float(real_presence.get("Breathing", 0.0)), 4)

    missing_in_synthetic = [
        cat
        for cat in FEATURE_ORDER
        if float(syn_presence.get(cat, 0.0)) < 0.5 and float(real_presence.get(cat, 0.0)) >= 0.9
    ]
    low_coverage_cats_synthetic = [
        cat for cat in FEATURE_ORDER if float(syn_presence.get(cat, 0.0)) < 0.9
    ]

    return {
        "synthetic_mean_category_coverage": syn_cov,
        "real_mean_category_coverage": real_cov,
        "coverage_delta_synthetic_minus_real": round(syn_cov - real_cov, 4),
        "breathing_presence_delta_synthetic_minus_real": breathing_gap,
        "categories_below_90pct_in_synthetic": low_coverage_cats_synthetic,
        "categories_well_in_real_but_weak_in_synthetic": missing_in_synthetic,
        "synthetic_all_7_slots_rate": synthetic.get("all_7_slots_rate", 0.0),
        "real_all_7_slots_rate": real.get("all_7_slots_rate", 0.0),
        "synthetic_duplicate_rate": synthetic["duplicates"]["duplicate_rate"],
        "real_duplicate_rate": real["duplicates"]["duplicate_rate"],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Synthetic vs real mFDA report audit (D1)",
        "",
        "## Summary",
        "",
    ]
    cmp = report.get("comparison", {})
    lines.extend(
        [
            f"- **Synthetic (train+val pooled) mean category coverage:** {cmp.get('synthetic_mean_category_coverage')}",
            f"- **Real test mean category coverage:** {cmp.get('real_mean_category_coverage')}",
            f"- **Coverage delta (synthetic − real):** {cmp.get('coverage_delta_synthetic_minus_real')}",
            f"- **All-7-slots rate — synthetic:** {cmp.get('synthetic_all_7_slots_rate')} | **real:** {cmp.get('real_all_7_slots_rate')}",
            f"- **Target duplicate rate — synthetic:** {cmp.get('synthetic_duplicate_rate')} | **real:** {cmp.get('real_duplicate_rate')}",
            f"- **Breathing presence delta (synthetic − real):** {cmp.get('breathing_presence_delta_synthetic_minus_real')}",
            "",
        ]
    )

    if cmp.get("categories_below_90pct_in_synthetic"):
        lines.append(
            f"- **Categories below 90% presence in synthetic:** {', '.join(cmp['categories_below_90pct_in_synthetic'])}"
        )
    if cmp.get("categories_well_in_real_but_weak_in_synthetic"):
        lines.append(
            "- **Real test has strong presence but synthetic is weak:** "
            + ", ".join(cmp["categories_well_in_real_but_weak_in_synthetic"])
        )
    lines.append("")

    for split in ("train", "val", "test"):
        block = report.get(split)
        if not block:
            continue
        lines.append(f"## {split.upper()} (n={block['n']}, is_real={block['is_real']})")
        lines.append("")
        lines.append(f"- Mean category coverage: **{block['mean_category_coverage']}**")
        lines.append(f"- All-7-slots rate: **{block['all_7_slots_rate']}**")
        lines.append(
            f"- Target length (chars): p50 **{block['target_len_chars']['p50']}**, "
            f"mean **{block['target_len_chars']['mean']}**"
        )
        lines.append(f"- Duplicate target rate: **{block['duplicates']['duplicate_rate']}**")
        lines.append("")
        lines.append("| Category | presence rate |")
        lines.append("|----------|---------------|")
        for cat in FEATURE_ORDER:
            rate = block["category_presence_rate"].get(cat, 0.0)
            lines.append(f"| {cat} | {rate} |")
        lines.append("")

        if "by_group" in block:
            lines.append("### By group (test)")
            for label, gblock in block["by_group"].items():
                lines.append(
                    f"- **{label}** (n={gblock['n']}): coverage **{gblock['mean_category_coverage']}**, "
                    f"len p50 **{gblock['target_len_chars']['p50']}**"
                )
            lines.append("")

    lines.append("## Interpretation hints")
    lines.append("")
    lines.append(
        "- If **synthetic coverage ≈ 1.0** but **model decode coverage ≈ 0.14**, the gap is likely "
        "generation/training dynamics, not missing structure in training targets."
    )
    lines.append(
        "- If **synthetic coverage is well below real test**, re-prepare or fix ETL/report templates before B23."
    )
    lines.append(
        "- High **duplicate_rate** on synthetic train/val suggests memorizable repeated phrases."
    )
    lines.append("")
    return "\n".join(lines)


def build_audit_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    max_duplicate_examples: int = 5,
) -> dict[str, Any]:
    train_audit = audit_dataframe(train_df, max_duplicate_examples=max_duplicate_examples)
    val_audit = audit_dataframe(val_df, max_duplicate_examples=max_duplicate_examples)
    test_audit = audit_dataframe(test_df, max_duplicate_examples=max_duplicate_examples)

    synthetic_texts = (
        train_df["target_text"].astype(str).tolist() + val_df["target_text"].astype(str).tolist()
    )
    synthetic_pooled = audit_dataframe(
        pd.DataFrame({"target_text": synthetic_texts, "is_real": [False] * len(synthetic_texts), "group": [None] * len(synthetic_texts)}),
        max_duplicate_examples=max_duplicate_examples,
    )

    comparison = _compare_splits(synthetic_pooled, test_audit)

    return {
        "train_csv": str(train_df["source_file"].iloc[0]) if len(train_df) else None,
        "val_csv": str(val_df["source_file"].iloc[0]) if len(val_df) else None,
        "test_csv": str(test_df["source_file"].iloc[0]) if len(test_df) else None,
        "train": train_audit,
        "val": val_audit,
        "test": test_audit,
        "synthetic_pooled": synthetic_pooled,
        "comparison": comparison,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    train_path = resolve_under_repo(args.train)
    val_path = resolve_under_repo(args.val)
    test_path = resolve_under_repo(args.test)
    for label, path in (("train", train_path), ("val", val_path), ("test", test_path)):
        if not path.is_file():
            raise SystemExit(f"{label} CSV not found: {path}")

    print(f"[audit_report_distribution] train = {train_path}", flush=True)
    print(f"[audit_report_distribution] val   = {val_path}", flush=True)
    print(f"[audit_report_distribution] test  = {test_path}", flush=True)

    train_df = _load_split_csv(train_path, "train", is_real=False)
    val_df = _load_split_csv(val_path, "val", is_real=False)
    test_df = _load_split_csv(test_path, "test", is_real=True)

    report = build_audit_report(
        train_df,
        val_df,
        test_df,
        max_duplicate_examples=args.sample_duplicate_examples,
    )

    out_json = resolve_under_repo(args.output_json or Path("analysis/synthetic_real_report_audit.json"))
    out_md = resolve_under_repo(args.output_md or Path("analysis/synthetic_real_report_audit.md"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"[audit_report_distribution] wrote {out_json}", flush=True)
    print(f"[audit_report_distribution] wrote {out_md}", flush=True)
    cmp = report["comparison"]
    print(
        "[audit_report_distribution] synthetic coverage "
        f"{cmp['synthetic_mean_category_coverage']} | real {cmp['real_mean_category_coverage']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
