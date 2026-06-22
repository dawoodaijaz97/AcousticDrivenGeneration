"""Shared mFDA report severity parsing (train oversampling, PD analysis)."""

from __future__ import annotations

import re

from main.prompts import FEATURE_ORDER

SEVERITIES: tuple[str, ...] = ("Normal", "Mild", "Moderate", "Severe")


def _category_pattern(category: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b{re.escape(category)}\s*\(\s*({'|'.join(SEVERITIES)})\s*\)\s*:",
        re.IGNORECASE,
    )


_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    cat: _category_pattern(cat) for cat in FEATURE_ORDER
}


def parse_severities(text: str) -> dict[str, str | None]:
    """Return severity label per mFDA category, or None if missing."""
    out: dict[str, str | None] = {}
    for cat in FEATURE_ORDER:
        m = _CATEGORY_PATTERNS[cat].search(text)
        if not m:
            out[cat] = None
            continue
        label = m.group(1)
        for sev in SEVERITIES:
            if label.lower() == sev.lower():
                out[cat] = sev
                break
        else:
            out[cat] = label
    return out


def target_meets_severity_min(target_text: str, min_label: str) -> bool:
    """True if any parsed category severity is >= min_label."""
    sevs = parse_severities(target_text)
    present = [s for s in sevs.values() if s in SEVERITIES]
    if not present:
        return False
    min_rank = SEVERITIES.index(min_label)
    return max(SEVERITIES.index(s) for s in present) >= min_rank
