"""Decode eval for B23 structured seven-label models.

Generates ``Category: Severity`` lines, scores label accuracy, verbalizes to prose,
and reports standard Table-4 metrics against ``reference_prose``.
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
from main.structured_report import aggregate_structure_metrics
from main.structured_targets import (
    aggregate_label_metrics,
    aggregate_seven_label_structure_metrics,
    prose_to_seven_label_target,
    seven_label_target_to_prose,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode B23 structured seven-label model; score labels + verbalized prose.",
    )
    p.add_argument("--tokenized-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--tokenizer-model", type=str, required=True)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-predictions-json", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--no-repeat-ngram-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bleu-tokenize", type=str, default="13a")
    p.add_argument("--bleu-lowercase", action="store_true")
    p.add_argument("--bertscore-lang", type=str, default="en")  # reports are English; es under-scores BERT
    p.add_argument("--bertscore-batch-size", type=int, default=32)
    return p.parse_args(argv)


def _log(msg: str) -> None:
    print(f"[eval_structured] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    if args.cpu_only and args.require_gpu:
        raise SystemExit("Choose at most one of --cpu-only and --require-gpu.")
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch.")

    model_path = resolve_under_repo(args.model_path)
    if not model_path.is_dir():
        raise SystemExit(f"model path not found: {model_path}")

    tokenized = _resolve_tokenized_dataset_dir(resolve_under_repo(args.tokenized_dir))
    raw = load_from_disk(str(tokenized))
    if "test" not in raw:
        raise SystemExit("DatasetDict must contain a 'test' split.")
    test_ds: Dataset = raw["test"]

    device = _pick_device(args.cpu_only)
    _log(f"tokenized_dir = {tokenized}")
    _log(f"model_path    = {model_path}")
    _log(f"device        = {device}")
    _log(f"test rows     = {len(test_ds):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else None
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), torch_dtype=dtype)
    model.eval()
    model.to(device)

    label_hyps: list[str] = []
    label_refs: list[str] = []
    prose_hyps: list[str] = []
    prose_refs: list[str] = []
    groups: list[str | None] = []
    sample_ids: list[str | None] = []

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": False,
    }
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    bs = max(1, args.batch_size)
    n = len(test_ds)
    with torch.no_grad():
        for start in range(0, n, bs):
            batch_rows = test_ds[start : start + bs]
            batch_len = len(batch_rows["input_ids"])
            features = [
                {"input_ids": batch_rows["input_ids"][i], "attention_mask": batch_rows["attention_mask"][i]}
                for i in range(batch_len)
            ]
            padded = collator(features)
            gen = model.generate(
                input_ids=padded["input_ids"].to(device),
                attention_mask=padded["attention_mask"].to(device),
                **gen_kwargs,
            )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for i, text in enumerate(decoded):
                idx = start + i
                if "reference_prose" in test_ds.column_names:
                    ref_prose = str(test_ds[idx]["reference_prose"]).strip()
                else:
                    ref_prose = _labels_to_text(test_ds[idx]["labels"], tokenizer)

                hyp_labels = text.strip()
                ref_labels = prose_to_seven_label_target(ref_prose)
                hyp_prose = seven_label_target_to_prose(hyp_labels)

                label_hyps.append(hyp_labels)
                label_refs.append(ref_labels)
                prose_hyps.append(hyp_prose)
                prose_refs.append(ref_prose)

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
        rows = []
        for i in range(len(label_hyps)):
            rows.append(
                {
                    "sample_id": sample_ids[i],
                    "group": groups[i],
                    "reference_prose": prose_refs[i],
                    "reference_labels": label_refs[i],
                    "hypothesis_labels": label_hyps[i],
                    "hypothesis_prose": prose_hyps[i],
                }
            )
        pred_path = resolve_under_repo(args.output_predictions_json)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(
            json.dumps(
                {
                    "model_path": str(model_path),
                    "tokenized_dir": str(tokenized),
                    "n": len(rows),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
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

    label_overall = aggregate_label_metrics(list(zip(label_refs, label_hyps, strict=True)))
    label_structure = aggregate_seven_label_structure_metrics(label_hyps, groups)
    prose_structure = aggregate_structure_metrics(prose_hyps, groups)

    report: dict[str, Any] = {
        "eval_structured": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
        },
        "tokenized_dir": str(tokenized),
        "model_path": str(model_path),
        "tokenizer_model": args.tokenizer_model,
        "label_metrics": label_overall,
        "label_structure_metrics": label_structure,
        "verbalized_structure_metrics": prose_structure,
        "verbalized_overall": _subset_metrics_paper_table(prose_hyps, prose_refs, **metric_kwargs),
    }

    if any(g is not None for g in groups):
        by_group: dict[str, Any] = {}
        for label in ("PD", "HC"):
            idxs = [i for i, g in enumerate(groups) if g == label]
            if not idxs:
                continue
            h = [prose_hyps[i] for i in idxs]
            r = [prose_refs[i] for i in idxs]
            by_group[label] = _subset_metrics_paper_table(h, r, **metric_kwargs)
            by_group[label]["label_exact_match_rate"] = aggregate_label_metrics(
                [(label_refs[i], label_hyps[i]) for i in idxs]
            )["exact_match_rate"]
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
