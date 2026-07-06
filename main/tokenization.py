from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from main.prompts import DEFAULT_TASK_PREFIX, format_t5_source


def add_t5_text_columns(
    ds: Dataset,
    *,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    prompt_style: str | None = None,
    input_column: str = "input_text",
    source_column: str = "source_text",
) -> Dataset:
    """Add ``source_text`` for T5 encoder input while keeping raw ``input_text``."""

    def _map(batch: dict[str, list]) -> dict[str, list]:
        sources = [
            format_t5_source(t, task_prefix=task_prefix, prompt_style=prompt_style)
            for t in batch[input_column]
        ]
        return {source_column: sources}

    return ds.map(_map, batched=True)


def tokenize_seq2seq(
    ds: Dataset,
    tokenizer: Any,
    *,
    source_column: str = "source_text",
    target_column: str = "target_text",
    max_source_length: int = 256,
    max_target_length: int = 512,
    padding: str = "max_length",
    passthrough_columns: tuple[str, ...] = (),
) -> Dataset:
    """Tokenize for T5 fine-tuning; pads labels and masks padding in loss with -100."""

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id for seq2seq tokenization.")

    def _tokenize(batch: dict[str, list]) -> dict[str, Any]:
        model_inputs = tokenizer(
            batch[source_column],
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )
        labels_tok = tokenizer(
            batch[target_column],
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )
        labels = labels_tok["input_ids"]
        masked = []
        for seq in labels:
            masked.append([(t if t != pad_id else -100) for t in seq])
        out: dict[str, Any] = {**model_inputs, "labels": masked}
        for col in passthrough_columns:
            if col in batch:
                out[col] = batch[col]
        return out

    remove_cols = [source_column, target_column]
    return ds.map(_tokenize, batched=True, remove_columns=remove_cols)


def dataframe_splits_to_dataset_dict(
    splits: dict[str, Any],
    *,
    task_prefix: str = DEFAULT_TASK_PREFIX,
    prompt_style: str | None = None,
    input_column: str = "input_text",
) -> DatasetDict:
    """Build a ``DatasetDict`` from pandas frames keyed by split name."""
    parts: dict[str, Dataset] = {}
    for name, df in splits.items():
        if input_column not in df.columns:
            raise ValueError(
                f"Split {name!r} missing column {input_column!r} required for encoder input."
            )
        ds = Dataset.from_pandas(df, preserve_index=False)
        parts[name] = add_t5_text_columns(
            ds,
            task_prefix=task_prefix,
            prompt_style=prompt_style,
            input_column=input_column,
        )
    return DatasetDict(parts)


def tokenize_dataset_dict(
    ddict: DatasetDict,
    tokenizer: Any,
    *,
    max_source_length: int = 256,
    max_target_length: int = 512,
    padding: str = "max_length",
    passthrough_columns: tuple[str, ...] = (
        "sample_id",
        "split",
        "is_real",
        "group",
        "example_hash",
        "target_text",
    ),
) -> DatasetDict:
    """Tokenize every split; optional metadata columns are copied through."""

    source_column = "source_text"
    target_column = "target_text"

    out: dict[str, Dataset] = {}
    for k, ds in ddict.items():
        keep = tuple(c for c in passthrough_columns if c in ds.column_names)
        if "reference_prose" in ds.column_names and "reference_prose" not in keep:
            keep = (*keep, "reference_prose")
        # ``source_text`` / ``target_text`` are encoded once; still copy ``target_text`` through.
        passthrough = tuple(c for c in keep if c not in (source_column, target_column))
        passthrough_for_map = passthrough
        if target_column in ds.column_names:
            passthrough_for_map = (target_column, *passthrough)
        cols = [source_column, target_column, *passthrough]
        subset = ds.select_columns(cols)
        out[k] = tokenize_seq2seq(
            subset,
            tokenizer,
            source_column=source_column,
            target_column=target_column,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            passthrough_columns=passthrough_for_map,
        )
    return DatasetDict(out)
