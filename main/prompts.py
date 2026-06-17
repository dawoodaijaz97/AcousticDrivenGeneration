from __future__ import annotations

import re

# Default prefix used for S0–S5, B0/B4, L0/L4 (Phase 0–1 runs).
DEFAULT_TASK_PREFIX = (
    "generate mFDA clinical speech report from acoustic biomarkers: "
)

# Flan-T5 fine-tune style prefix (paper / Phase 2 — experiment B5).
FLAN_PAPER_TASK_PREFIX = "Generate a report for:"

# Flan-T5 Phase 2 variant (B6): explicit mFDA category hints in prefix.
FLAN_PAPER_CATEGORIES_TASK_PREFIX = (
    "Generate a report for the following mFDA categories: "
    "breathing, lips, larynx, palate, monotonicity, tongue, intelligibility."
)

# Flan-T5 Phase 2 variant (B7): paper prefix + deterministic numeric key/value format.
FLAN_PAPER_NUMERIC_LABELS_TASK_PREFIX = (
    "Generate a report using these category scores "
    "(0 to 100; higher means more severe):"
)

# Flan-T5 Phase 2 variant (B19): explicit seven-slot output template (PD structure lever).
FLAN_PAPER_REPORT_TEMPLATE_TASK_PREFIX = (
    "Generate an mFDA report. Each category must appear as "
    "Category (Normal|Mild|Moderate|Severe): description. "
    "Order: Breathing, Lips, Palate, Larynx, Monotonicity, Tongue, Intelligibility. "
    "Biomarkers:"
)

FEATURE_ORDER: tuple[str, ...] = (
    "Breathing",
    "Lips",
    "Palate",
    "Larynx",
    "Monotonicity",
    "Tongue",
    "Intelligibility",
)

PROMPT_STYLES: dict[str, str] = {
    "default": DEFAULT_TASK_PREFIX,
    "flan-paper": FLAN_PAPER_TASK_PREFIX,
    "flan-paper-categories": FLAN_PAPER_CATEGORIES_TASK_PREFIX,
    "flan-paper-numeric-labels": FLAN_PAPER_NUMERIC_LABELS_TASK_PREFIX,
    "flan-paper-report-template": FLAN_PAPER_REPORT_TEMPLATE_TASK_PREFIX,
}


def resolve_prompt_style(style: str) -> str:
    """Map CLI ``--prompt-style`` name to the task prefix string."""
    key = style.strip().lower()
    if key not in PROMPT_STYLES:
        allowed = ", ".join(sorted(PROMPT_STYLES))
        raise ValueError(f"Unknown prompt style {style!r}; choose one of: {allowed}")
    return PROMPT_STYLES[key]


def _extract_feature_values(input_text: str) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for label in FEATURE_ORDER:
        match = re.search(rf"\b{re.escape(label)}\s*:\s*([-+]?\d+(?:\.\d+)?)", input_text)
        if not match:
            return None
        values[label] = float(match.group(1))
    return values


def _format_numeric_label_features(input_text: str) -> str:
    values = _extract_feature_values(input_text)
    if values is None:
        # Fallback: keep original text if parser misses expected keys.
        return input_text.strip()
    parts = [f"{label}={values[label]:.2f}" for label in FEATURE_ORDER]
    return " ; ".join(parts)


def format_t5_source(
    input_text: str,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    *,
    prompt_style: str | None = None,
) -> str:
    """Wrap normalized feature string as the encoder input for T5-style models."""
    text = input_text.strip()
    style = (prompt_style or "").strip().lower()
    if style == "flan-paper-numeric-labels":
        text = _format_numeric_label_features(text)
    prefix = task_prefix.strip()
    if not prefix.endswith(" "):
        prefix = prefix + " "
    return f"{prefix}{text}"
