"""Structured decode eval: B21 forced seven-slot template, B22 structure-aware reranking.

Extends ``main.eval_decode`` with post-processing and structure metrics
(``category_coverage``, ``severity_word_coverage``, ``all_7_slots_rate``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding

from main.eval_decode import (
    _labels_to_text,
    _load_evaluate_metrics,
    _pick_device,
    _resolve_tokenized_dataset_dir,
    _subset_metrics_paper_table,
)
from main.paths import resolve_under_repo
from main.structured_report import (
    aggregate_structure_metrics,
    force_seven_slot_template as apply_seven_slot_template,
    pick_best_candidate,
    structure_metrics_for_text,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Decode test split with optional seven-slot forcing or structure-aware reranking "
            "(B21/B22), plus structure metrics."
        ),
    )
    p.add_argument(
        "--tokenized-dir",
        type=Path,
        required=True,
        help="DatasetDict from prepare --tokenize (must contain a test split).",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Fine-tuned checkpoint dir (e.g. B17 final_model).",
    )
    p.add_argument(
        "--tokenizer-model",
        type=str,
        required=True,
        help="HF model id used to load the tokenizer (and special tokens).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON (repo-relative if not absolute).",
    )
    p.add_argument(
        "--output-predictions-json",
        type=Path,
        default=None,
        help="Optional path to write per-example reference/hypothesis JSON.",
    )
    p.add_argument(
        "--force-seven-slot-template",
        action="store_true",
        help="B21: post-process each decode into a mandatory seven-slot skeleton.",
    )
    p.add_argument(
        "--structure-rerank",
        action="store_true",
        help="B22: generate multiple candidates and pick the best by structure score.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Beam hypotheses per input (use 3–5 with --structure-rerank).",
    )
    p.add_argument("--no-repeat-ngram-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit if no CUDA GPU is visible to PyTorch.",
    )
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bleu-tokenize", type=str, default="13a")
    p.add_argument("--bleu-lowercase", action="store_true")
    p.add_argument("--bertscore-lang", type=str, default="en")  # reports are English; es under-scores BERT
    p.add_argument("--bertscore-batch-size", type=int, default=32)
    return p.parse_args(argv)


def _log(msg: str) -> None:
    print(f"[structured_decode] {msg}", flush=True)


def _validate_generation_args(args: argparse.Namespace) -> None:
    if args.structure_rerank and args.num_return_sequences <= 1:
        args.num_return_sequences = 5
        _log("structure-rerank enabled: defaulting --num-return-sequences to 5")
    if args.num_return_sequences > 1 and args.num_beams < args.num_return_sequences:
        args.num_beams = max(8, args.num_return_sequences)
        _log(
            f"num_return_sequences={args.num_return_sequences}: "
            f"raising --num-beams to {args.num_beams}"
        )
    if args.num_beams < 1:
        raise SystemExit("--num-beams must be >= 1")
    if args.num_return_sequences < 1:
        raise SystemExit("--num-return-sequences must be >= 1")
    if args.num_return_sequences > args.num_beams:
        raise SystemExit("--num-return-sequences cannot exceed --num-beams")


def _decode_batch(
    model: Any,
    tokenizer: Any,
    collator: DataCollatorWithPadding,
    batch_rows: dict[str, list[Any]],
    device: torch.device,
    *,
    max_new_tokens: int,
    num_beams: int,
    num_return_sequences: int,
    no_repeat_ngram_size: int,
) -> list[list[str]]:
    batch_len = len(batch_rows["input_ids"])
    features = [
        {"input_ids": batch_rows["input_ids"][i], "attention_mask": batch_rows["attention_mask"][i]}
        for i in range(batch_len)
    ]
    padded = collator(features)
    input_ids = padded["input_ids"].to(device)
    attention_mask = padded["attention_mask"].to(device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": False,
    }
    if no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if num_return_sequences > 1:
        gen_kwargs["num_return_sequences"] = num_return_sequences
        gen_kwargs["early_stopping"] = True

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )
    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

    if num_return_sequences == 1:
        return [[text.strip()] for text in decoded]

    grouped: list[list[str]] = []
    for i in range(batch_len):
        start = i * num_return_sequences
        end = start + num_return_sequences
        grouped.append([text.strip() for text in decoded[start:end]])
    return grouped


def _finalize_hypothesis(
    candidates: list[str],
    *,
    structure_rerank: bool,
    force_seven_slot_template: bool,
) -> tuple[str, str | None, int | None, dict[str, float | bool]]:
    raw: str | None = None
    rerank_index: int | None = None
    if structure_rerank:
        rerank_index, chosen, struct = pick_best_candidate(candidates)
        raw = chosen
    else:
        chosen = candidates[0]
        struct = structure_metrics_for_text(chosen)

    if force_seven_slot_template:
        if raw is None:
            raw = chosen
        chosen = apply_seven_slot_template(raw)
        struct = structure_metrics_for_text(chosen)

    return chosen, raw, rerank_index, struct


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_generation_args(args)
    torch.manual_seed(args.seed)

    if args.cpu_only and args.require_gpu:
        raise SystemExit("Choose at most one of --cpu-only and --require-gpu.")
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch (torch.cuda.is_available() is False). "
            "Omit --require-gpu only if you intend to decode on CPU."
        )

    model_path = resolve_under_repo(args.model_path)
    if not model_path.is_dir():
        raise SystemExit(f"model path not found: {model_path}")

    tokenized = _resolve_tokenized_dataset_dir(resolve_under_repo(args.tokenized_dir))
    raw = load_from_disk(str(tokenized))
    if "test" not in raw:
        raise SystemExit("DatasetDict must contain a 'test' split for decode metrics.")
    test_ds: Dataset = raw["test"]

    required = {"input_ids", "attention_mask", "labels"}
    missing = required - set(test_ds.column_names)
    if missing:
        raise SystemExit(f"test split missing columns {sorted(missing)}; got {test_ds.column_names}")

    device = _pick_device(args.cpu_only)
    _log(f"repo tokenized_dir = {tokenized}")
    _log(f"model_path         = {model_path}")
    _log(f"device             = {device}")
    _log(f"test rows          = {len(test_ds):,}")
    _log(f"force_template     = {args.force_seven_slot_template}")
    _log(f"structure_rerank   = {args.structure_rerank}")
    _log(f"num_beams          = {args.num_beams}")
    _log(f"num_return_seqs    = {args.num_return_sequences}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else None
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), torch_dtype=dtype)
    model.eval()
    model.to(device)

    hypotheses: list[str] = []
    raw_hypotheses: list[str | None] = []
    references: list[str] = []
    groups: list[str | None] = []
    sample_ids: list[str | None] = []
    rerank_indices: list[int | None] = []
    per_row_structure: list[dict[str, float | bool]] = []

    raw_hypotheses_for_metrics: list[str] = []
    has_raw_for_metrics = False

    bs = max(1, args.batch_size)
    n = len(test_ds)
    with torch.no_grad():
        for start in range(0, n, bs):
            batch_rows = test_ds[start : start + bs]
            candidate_groups = _decode_batch(
                model,
                tokenizer,
                collator,
                batch_rows,
                device,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

            batch_len = len(batch_rows["input_ids"])
            for i in range(batch_len):
                idx = start + i
                label_ids = test_ds[idx]["labels"]
                ref = _labels_to_text(label_ids, tokenizer)
                hyp, raw_hyp, rerank_idx, struct = _finalize_hypothesis(
                    candidate_groups[i],
                    structure_rerank=args.structure_rerank,
                    force_seven_slot_template=args.force_seven_slot_template,
                )

                hypotheses.append(hyp)
                raw_hypotheses.append(raw_hyp)
                references.append(ref)
                rerank_indices.append(rerank_idx)
                per_row_structure.append(struct)

                if raw_hyp is not None:
                    raw_hypotheses_for_metrics.append(raw_hyp)
                    has_raw_for_metrics = True
                elif args.structure_rerank:
                    raw_hypotheses_for_metrics.append(candidate_groups[i][0])
                    has_raw_for_metrics = True

                if "group" in test_ds.column_names:
                    g = test_ds[idx]["group"]
                    groups.append(None if g is None else str(g))
                else:
                    groups.append(None)
                if "sample_id" in test_ds.column_names:
                    sid = test_ds[idx]["sample_id"]
                    sample_ids.append(None if sid is None else str(sid))
                else:
                    sample_ids.append(None)

    if args.output_predictions_json is not None:
        pred_rows = []
        for i in range(len(hypotheses)):
            row: dict[str, Any] = {
                "sample_id": sample_ids[i] if i < len(sample_ids) else None,
                "group": groups[i] if i < len(groups) else None,
                "reference": references[i],
                "hypothesis": hypotheses[i],
            }
            if raw_hypotheses[i] is not None:
                row["raw_hypothesis"] = raw_hypotheses[i]
            if rerank_indices[i] is not None:
                row["rerank_candidate_index"] = rerank_indices[i]
            row["structure_metrics"] = per_row_structure[i]
            pred_rows.append(row)

        pred_report = {
            "model_path": str(model_path),
            "tokenized_dir": str(tokenized),
            "structured_decode": {
                "force_seven_slot_template": args.force_seven_slot_template,
                "structure_rerank": args.structure_rerank,
                "num_beams": args.num_beams,
                "num_return_sequences": args.num_return_sequences,
            },
            "n": len(pred_rows),
            "rows": pred_rows,
        }
        pred_path = resolve_under_repo(args.output_predictions_json)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps(pred_report, indent=2), encoding="utf-8")
        _log(f"wrote predictions {pred_path}")

    bert_device = "cuda" if device.type == "cuda" else "cpu"
    rouge_m, bleu_m, bert_m = _load_evaluate_metrics()

    metric_kwargs = {
        "rouge_m": rouge_m,
        "bleu_m": bleu_m,
        "bert_m": bert_m,
        "bleu_tokenize": args.bleu_tokenize,
        "bleu_lowercase": args.bleu_lowercase,
        "bertscore_lang": args.bertscore_lang,
        "bertscore_batch_size": args.bertscore_batch_size,
        "bert_device": bert_device,
    }

    overall = _subset_metrics_paper_table(hypotheses, references, **metric_kwargs)
    structure_metrics = aggregate_structure_metrics(hypotheses, groups)

    report: dict[str, Any] = {
        "structured_decode": {
            "force_seven_slot_template": args.force_seven_slot_template,
            "structure_rerank": args.structure_rerank,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "max_new_tokens": args.max_new_tokens,
            "no_repeat_ngram_size": args.no_repeat_ngram_size if args.no_repeat_ngram_size > 0 else None,
            "batch_size": args.batch_size,
        },
        "tokenized_dir": str(tokenized),
        "model_path": str(model_path),
        "tokenizer_model": args.tokenizer_model,
        "structure_metrics": structure_metrics,
        "overall": overall,
    }

    if has_raw_for_metrics and (
        args.force_seven_slot_template or args.structure_rerank
    ):
        report["overall_raw"] = _subset_metrics_paper_table(
            raw_hypotheses_for_metrics,
            references,
            **metric_kwargs,
        )
        report["structure_metrics_raw"] = aggregate_structure_metrics(
            raw_hypotheses_for_metrics,
            groups,
        )

    if any(g is not None for g in groups):
        by_group: dict[str, dict[str, Any]] = {}
        for label in ("PD", "HC"):
            idxs = [i for i, g in enumerate(groups) if g == label]
            if not idxs:
                continue
            h = [hypotheses[i] for i in idxs]
            r = [references[i] for i in idxs]
            by_group[label] = _subset_metrics_paper_table(h, r, **metric_kwargs)
        report["by_group"] = by_group

    text = json.dumps(report, indent=2)
    if args.output_json is not None:
        out_path = resolve_under_repo(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        _log(f"wrote {out_path}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
