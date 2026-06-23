"""Seven-slot mFDA report structure: parse, force template, score, aggregate metrics."""

from __future__ import annotations

import re
from typing import Any

from main.prompts import FEATURE_ORDER
from main.report_severity import SEVERITIES, parse_severities

DEFAULT_MISSING_SEVERITY = "Normal"
DEFAULT_MISSING_DESCRIPTION = "Within normal limits."

_SLOT_HEADER_RE = re.compile(
    rf"(?P<cat>{'|'.join(re.escape(c) for c in FEATURE_ORDER)})"
    rf"\s*\(\s*(?P<sev>{'|'.join(SEVERITIES)})\s*\)\s*:\s*",
    re.IGNORECASE,
)


def _canonical_severity(label: str) -> str:
    for sev in SEVERITIES:
        if label.lower() == sev.lower():
            return sev
    return label


def parse_slot_descriptions(text: str) -> dict[str, tuple[str | None, str | None]]:
    """Return ``category -> (severity, description)`` parsed from slot headers."""
    matches = list(_SLOT_HEADER_RE.finditer(text))
    out: dict[str, tuple[str | None, str | None]] = {cat: (None, None) for cat in FEATURE_ORDER}

    for i, match in enumerate(matches):
        cat_raw = match.group("cat")
        cat = next(c for c in FEATURE_ORDER if c.lower() == cat_raw.lower())
        sev = _canonical_severity(match.group("sev"))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        desc = text[start:end].strip() or None
        out[cat] = (sev, desc)

    # Fill severities found by the shared parser even if description extraction missed.
    for cat, sev in parse_severities(text).items():
        cur_sev, cur_desc = out[cat]
        if cur_sev is None and sev is not None:
            out[cat] = (sev, cur_desc)

    return out


def force_seven_slot_template(
    raw_text: str,
    *,
    default_severity: str = DEFAULT_MISSING_SEVERITY,
    default_description: str = DEFAULT_MISSING_DESCRIPTION,
) -> str:
    """Normalize free-form decode into a mandatory seven-category skeleton."""
    slots = parse_slot_descriptions(raw_text)
    lines: list[str] = []
    for cat in FEATURE_ORDER:
        sev, desc = slots[cat]
        if sev is None:
            sev = default_severity
        if not desc:
            desc = default_description
        lines.append(f"{cat} ({sev}): {desc}")
    return "\n".join(lines)


def structure_metrics_for_text(text: str) -> dict[str, float | bool]:
    """Per-example structure metrics used for B21/B22 promotion gates."""
    sevs = parse_severities(text)
    n = len(FEATURE_ORDER)
    present = sum(1 for v in sevs.values() if v is not None)
    valid = sum(1 for v in sevs.values() if v in SEVERITIES)
    return {
        "category_coverage": present / n,
        "severity_word_coverage": valid / n,
        "all_7_slots": present == n,
    }


def pick_best_candidate(candidates: list[str]) -> tuple[int, str, dict[str, float | bool]]:
    """Select the candidate with the strongest category/severity structure."""
    if not candidates:
        raise ValueError("pick_best_candidate requires at least one candidate")
    best_idx = 0
    best_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    best_metrics: dict[str, float | bool] = structure_metrics_for_text(candidates[0])
    for i, cand in enumerate(candidates):
        metrics = structure_metrics_for_text(cand)
        key = (
            float(metrics["all_7_slots"]),
            float(metrics["category_coverage"]),
            float(metrics["severity_word_coverage"]),
        )
        if key > best_key:
            best_key = key
            best_idx = i
            best_metrics = metrics
    return best_idx, candidates[best_idx], best_metrics


def aggregate_structure_metrics(
    texts: list[str],
    groups: list[str | None] | None = None,
) -> dict[str, Any]:
    """Corpus-level structure metrics for decode JSON output."""
    if not texts:
        return {
            "n": 0,
            "category_coverage": 0.0,
            "severity_word_coverage": 0.0,
            "all_7_slots_rate": 0.0,
        }

    per_example = [structure_metrics_for_text(t) for t in texts]
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
            subset = [texts[i] for i in idxs]
            by_group[label] = aggregate_structure_metrics(subset)
        report["by_group"] = by_group

    return report
