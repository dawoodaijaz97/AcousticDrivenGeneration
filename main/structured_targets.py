"""B23 structured seven-label targets: convert, parse, verbalize, score."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from main.prompts import FEATURE_ORDER
from main.report_severity import SEVERITIES, parse_severities
from main.structured_report import DEFAULT_MISSING_DESCRIPTION, DEFAULT_MISSING_SEVERITY

_SEVEN_LABEL_LINE_RE = re.compile(
    rf"^(?P<cat>{'|'.join(re.escape(c) for c in FEATURE_ORDER)})"
    rf"\s*:\s*(?P<sev>{'|'.join(SEVERITIES)})\s*$",
    re.IGNORECASE,
)


def _canonical_severity(label: str) -> str:
    for sev in SEVERITIES:
        if label.lower() == sev.lower():
            return sev
    return label


def prose_to_seven_label_target(prose: str, *, default_severity: str = DEFAULT_MISSING_SEVERITY) -> str:
    """Convert full clinical prose into one ``Category: Severity`` line per mFDA slot."""
    sevs = parse_severities(prose)
    lines: list[str] = []
    for cat in FEATURE_ORDER:
        sev = sevs.get(cat) or default_severity
        if sev not in SEVERITIES:
            sev = default_severity
        lines.append(f"{cat}: {sev}")
    return "\n".join(lines)


def parse_seven_label_target(text: str) -> dict[str, str | None]:
    """Parse ``Category: Severity`` lines; missing categories are ``None``."""
    out: dict[str, str | None] = {cat: None for cat in FEATURE_ORDER}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SEVEN_LABEL_LINE_RE.match(line)
        if not m:
            continue
        cat_raw = m.group("cat")
        cat = next(c for c in FEATURE_ORDER if c.lower() == cat_raw.lower())
        out[cat] = _canonical_severity(m.group("sev"))
    return out


def seven_label_target_to_prose(
    seven_label: str,
    *,
    default_severity: str = DEFAULT_MISSING_SEVERITY,
    default_description: str = DEFAULT_MISSING_DESCRIPTION,
) -> str:
    """Deterministic verbalization: seven-label target → ``Category (Severity): …`` prose."""
    labels = parse_seven_label_target(seven_label)
    lines: list[str] = []
    for cat in FEATURE_ORDER:
        sev = labels.get(cat) or default_severity
        if sev not in SEVERITIES:
            sev = default_severity
        lines.append(f"{cat} ({sev}): {default_description}")
    return "\n".join(lines)


def apply_structured_target_format(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Keep original prose in ``reference_prose``; train on ``target_text`` seven-label strings."""
    out: dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        frame = df.copy()
        frame["reference_prose"] = frame["target_text"].astype(str)
        frame["target_text"] = frame["reference_prose"].map(prose_to_seven_label_target)
        out[name] = frame
    return out


def label_metrics_for_pair(reference_labels: str, hypothesis_labels: str) -> dict[str, Any]:
    """Exact-match severity accuracy per category and macro average."""
    ref = parse_seven_label_target(reference_labels)
    hyp = parse_seven_label_target(hypothesis_labels)
    per_category: dict[str, bool | None] = {}
    matches = 0
    comparable = 0
    for cat in FEATURE_ORDER:
        r, h = ref[cat], hyp[cat]
        if r is None:
            per_category[cat] = None
            continue
        comparable += 1
        ok = h == r
        per_category[cat] = ok
        if ok:
            matches += 1
    return {
        "comparable_slots": comparable,
        "exact_match_rate": matches / comparable if comparable else 0.0,
        "per_category_exact_match": per_category,
    }


def aggregate_seven_label_structure_metrics(
    texts: list[str],
    groups: list[str | None] | None = None,
) -> dict[str, Any]:
    """Structure metrics for ``Category: Severity`` label targets (B23)."""
    if not texts:
        return {
            "n": 0,
            "category_coverage": 0.0,
            "severity_word_coverage": 0.0,
            "all_7_slots_rate": 0.0,
        }

    per_example: list[dict[str, float | bool]] = []
    n_cats = len(FEATURE_ORDER)
    for text in texts:
        parsed = parse_seven_label_target(text)
        present = sum(1 for v in parsed.values() if v is not None)
        valid = sum(1 for v in parsed.values() if v in SEVERITIES)
        per_example.append(
            {
                "category_coverage": present / n_cats,
                "severity_word_coverage": valid / n_cats,
                "all_7_slots": present == n_cats,
            }
        )

    n = len(per_example)

    def _mean(field: str) -> float:
        return sum(float(m[field]) for m in per_example) / n

    report: dict[str, Any] = {
        "n": n,
        "category_coverage": round(_mean("category_coverage"), 4),
        "severity_word_coverage": round(_mean("severity_word_coverage"), 4),
        "all_7_slots_rate": round(sum(1 for m in per_example if m["all_7_slots"]) / n, 4),
    }

    if groups is not None and any(g is not None for g in groups):
        by_group: dict[str, Any] = {}
        for label in ("PD", "HC"):
            idxs = [i for i, g in enumerate(groups) if g == label]
            if not idxs:
                continue
            by_group[label] = aggregate_seven_label_structure_metrics([texts[i] for i in idxs])
        report["by_group"] = by_group

    return report


def aggregate_label_metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "exact_match_rate": 0.0, "per_category_exact_match_rate": {}}

    per_cat_hits = {cat: 0 for cat in FEATURE_ORDER}
    per_cat_total = {cat: 0 for cat in FEATURE_ORDER}
    rates: list[float] = []

    for ref, hyp in pairs:
        m = label_metrics_for_pair(ref, hyp)
        rates.append(float(m["exact_match_rate"]))
        for cat, ok in m["per_category_exact_match"].items():
            if ok is None:
                continue
            per_cat_total[cat] += 1
            if ok:
                per_cat_hits[cat] += 1

    return {
        "n": len(pairs),
        "exact_match_rate": round(sum(rates) / len(rates), 4),
        "per_category_exact_match_rate": {
            cat: round(per_cat_hits[cat] / per_cat_total[cat], 4) if per_cat_total[cat] else 0.0
            for cat in FEATURE_ORDER
        },
    }
