"""Per-category / PD-vs-HC analysis on decoded mFDA reports.

Reads ``test_decode_predictions.json`` from ``main.eval_decode --output-predictions-json``
and reports category coverage, severity-label match rates, and PD vs HC gaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from main.paths import resolve_under_repo
from main.prompts import FEATURE_ORDER
from main.report_severity import SEVERITIES, parse_severities


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze mFDA category coverage on decode predictions.")
    p.add_argument(
        "--predictions-json",
        type=Path,
        required=True,
        help="JSON from main.eval_decode --output-predictions-json",
    )
    p.add_argument(
        "--baseline-predictions-json",
        type=Path,
        default=None,
        help="Optional second predictions file (e.g. B11) for comparison.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write analysis summary JSON (default: alongside predictions as pd_analysis.json).",
    )
    return p.parse_args(argv)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_cats = len(FEATURE_ORDER)

    coverage_ref: list[float] = []
    coverage_hyp: list[float] = []
    severity_match: list[float] = []
    by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"coverage_hyp": [], "coverage_ref": [], "severity_match": []}
    )
    per_category: dict[str, dict[str, Any]] = {
        cat: {
            "present_in_ref": 0,
            "present_in_hyp": 0,
            "severity_match": 0,
            "severity_mismatch_examples": [],
        }
        for cat in FEATURE_ORDER
    }
    weak_pd: list[dict[str, Any]] = []

    for row in rows:
        ref = str(row.get("reference") or "")
        hyp = str(row.get("hypothesis") or "")
        group = row.get("group")
        sample_id = row.get("sample_id")

        ref_sev = parse_severities(ref)
        hyp_sev = parse_severities(hyp)

        ref_cov = sum(1 for v in ref_sev.values() if v is not None) / n_cats
        hyp_cov = sum(1 for v in hyp_sev.values() if v is not None) / n_cats
        coverage_ref.append(ref_cov)
        coverage_hyp.append(hyp_cov)

        matches = 0
        comparable = 0
        for cat in FEATURE_ORDER:
            r = ref_sev[cat]
            h = hyp_sev[cat]
            if r is not None:
                per_category[cat]["present_in_ref"] += 1
            if h is not None:
                per_category[cat]["present_in_hyp"] += 1
            if r is not None and h is not None:
                comparable += 1
                if r == h:
                    matches += 1
                    per_category[cat]["severity_match"] += 1
                elif group == "PD" and len(per_category[cat]["severity_mismatch_examples"]) < 3:
                    per_category[cat]["severity_mismatch_examples"].append(
                        {
                            "sample_id": sample_id,
                            "ref": r,
                            "hyp": h,
                        }
                    )
        sev_rate = matches / comparable if comparable else 0.0
        severity_match.append(sev_rate)

        if group in ("PD", "HC"):
            by_group[group]["coverage_hyp"].append(hyp_cov)
            by_group[group]["coverage_ref"].append(ref_cov)
            by_group[group]["severity_match"].append(sev_rate)

        if group == "PD":
            weak_pd.append(
                {
                    "sample_id": sample_id,
                    "coverage_hyp": round(hyp_cov, 4),
                    "severity_match": round(sev_rate, 4),
                    "missing_categories": [c for c in FEATURE_ORDER if hyp_sev[c] is None],
                }
            )

    weak_pd.sort(key=lambda x: (x["coverage_hyp"], x["severity_match"]))

    per_cat_summary: dict[str, Any] = {}
    for cat in FEATURE_ORDER:
        block = per_category[cat]
        ref_n = block["present_in_ref"] or 1
        hyp_n = max(block["present_in_hyp"], 1)
        per_cat_summary[cat] = {
            "ref_presence_rate": round(block["present_in_ref"] / n, 4),
            "hyp_presence_rate": round(block["present_in_hyp"] / n, 4),
            "severity_match_rate": round(block["severity_match"] / ref_n, 4),
            "pd_mismatch_examples": block["severity_mismatch_examples"],
        }

    group_summary: dict[str, Any] = {}
    for label in ("PD", "HC"):
        g = by_group.get(label)
        if not g:
            continue
        group_summary[label] = {
            "n": len(g["coverage_hyp"]),
            "mean_coverage_hyp": round(_mean(g["coverage_hyp"]) or 0.0, 4),
            "mean_coverage_ref": round(_mean(g["coverage_ref"]) or 0.0, 4),
            "mean_severity_match": round(_mean(g["severity_match"]) or 0.0, 4),
        }

    return {
        "n": n,
        "mean_coverage_hyp": round(_mean(coverage_hyp) or 0.0, 4),
        "mean_coverage_ref": round(_mean(coverage_ref) or 0.0, 4),
        "mean_severity_match": round(_mean(severity_match) or 0.0, 4),
        "by_group": group_summary,
        "per_category": per_cat_summary,
        "weakest_pd_examples": weak_pd[:10],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pred_path = resolve_under_repo(args.predictions_json)
    data = json.loads(pred_path.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    if not rows:
        print("No rows in predictions JSON.", file=sys.stderr)
        return 1

    report: dict[str, Any] = {
        "predictions_json": str(pred_path),
        "model_path": data.get("model_path"),
        "analysis": analyze_rows(rows),
    }

    if args.baseline_predictions_json is not None:
        base_path = resolve_under_repo(args.baseline_predictions_json)
        base_data = json.loads(base_path.read_text(encoding="utf-8"))
        base_rows = base_data.get("rows") or []
        report["baseline_predictions_json"] = str(base_path)
        report["baseline_model_path"] = base_data.get("model_path")
        report["baseline_analysis"] = analyze_rows(base_rows)
        cur = report["analysis"]["by_group"]
        base = report["baseline_analysis"]["by_group"]
        deltas: dict[str, Any] = {}
        for label in ("PD", "HC"):
            if label in cur and label in base:
                deltas[label] = {
                    "coverage_hyp_delta": round(
                        cur[label]["mean_coverage_hyp"] - base[label]["mean_coverage_hyp"], 4
                    ),
                    "severity_match_delta": round(
                        cur[label]["mean_severity_match"] - base[label]["mean_severity_match"], 4
                    ),
                }
        report["baseline_deltas"] = deltas

    out_path = args.output_json
    if out_path is None:
        out_path = pred_path.with_name("pd_analysis.json")
    out_path = resolve_under_repo(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
