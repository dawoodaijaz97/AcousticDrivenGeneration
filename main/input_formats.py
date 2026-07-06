from __future__ import annotations

# Column on ETL frames used as the encoder feature string before the task prefix.
INPUT_FORMAT_COLUMNS: dict[str, str] = {
    "compact-seven": "input_text",
    "instructions-raw": "instructions_raw",
}


def resolve_input_format(name: str) -> tuple[str, str]:
    """Return ``(format_key, dataframe_column)`` for ``--input-format``."""
    key = name.strip().lower()
    if key not in INPUT_FORMAT_COLUMNS:
        allowed = ", ".join(sorted(INPUT_FORMAT_COLUMNS))
        raise ValueError(f"Unknown input format {name!r}; choose one of: {allowed}")
    return key, INPUT_FORMAT_COLUMNS[key]
