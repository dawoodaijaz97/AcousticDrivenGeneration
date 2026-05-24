from __future__ import annotations

# Default prefix used for S0–S5, B0/B4, L0/L4 (Phase 0–1 runs).
DEFAULT_TASK_PREFIX = (
    "generate mFDA clinical speech report from acoustic biomarkers: "
)

# Flan-T5 fine-tune style prefix (paper / Phase 2 — experiment B5).
FLAN_PAPER_TASK_PREFIX = "Generate a report for:"

PROMPT_STYLES: dict[str, str] = {
    "default": DEFAULT_TASK_PREFIX,
    "flan-paper": FLAN_PAPER_TASK_PREFIX,
}


def resolve_prompt_style(style: str) -> str:
    """Map CLI ``--prompt-style`` name to the task prefix string."""
    key = style.strip().lower()
    if key not in PROMPT_STYLES:
        allowed = ", ".join(sorted(PROMPT_STYLES))
        raise ValueError(f"Unknown prompt style {style!r}; choose one of: {allowed}")
    return PROMPT_STYLES[key]


def format_t5_source(input_text: str, task_prefix: str = DEFAULT_TASK_PREFIX) -> str:
    """Wrap normalized feature string as the encoder input for T5-style models."""
    text = input_text.strip()
    prefix = task_prefix.strip()
    if not prefix.endswith(" "):
        prefix = prefix + " "
    return f"{prefix}{text}"
