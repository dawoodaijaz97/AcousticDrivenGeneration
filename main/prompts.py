from __future__ import annotations

# Stable prefix for instruction-tuned / seq2seq models (T5 family).
DEFAULT_TASK_PREFIX = (
    "generate mFDA clinical speech report from acoustic biomarkers: "
)


def format_t5_source(input_text: str, task_prefix: str = DEFAULT_TASK_PREFIX) -> str:
    """Wrap normalized feature string as the encoder input for T5-style models."""
    text = input_text.strip()
    prefix = task_prefix.strip()
    if not prefix.endswith(" "):
        prefix = prefix + " "
    return f"{prefix}{text}"
