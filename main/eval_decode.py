"""Decode the real test split and score with the same metrics as the paper (Table 4).

Paper: *Acoustic-Driven Generation of Pathological Speech Reports Using Large Language Models*
(Arias-Vergara et al., posted Aug 2025, DOI rs.3.rs-7326708/v1), ``docs/paper.pdf``.

**Generation (Section 3, page 7)** — match inference settings reported there:

- **512** output tokens (``max_new_tokens`` here).
- **Beam search: 3** (``--num-beams``, default 3).
- **No repeating n-gram: 2** (``--no-repeat-ngram-size``, default 2).

**Metrics (Section 3 + Table 4)** — ROUGE-1 / ROUGE-2 / ROUGE-L, corpus BLEU, and BERT scores as in the table.
Reference [40] is a survey of MT evaluation metrics (no implementation detail). To maximize comparability with
common HF-based workflows, this script uses HuggingFace ``evaluate`` wrappers:

- **BLEU**: ``evaluate``'s SacreBLEU — corpus BLEU-4, ``smooth_method='exp'``, default tokenizer **13a**
  (``BLEU.TOKENIZER_DEFAULT`` in SacreBLEU; same as ``evaluate.load('sacrebleu')`` when ``tokenize`` is omitted).
  Reported as **BLEU** on the 0–1 scale (SacreBLEU score / 100), consistent with Table 4 (e.g. 0.732).
- **ROUGE**: ``evaluate``'s ROUGE — same ``rouge_score`` backend as the metric card; **bootstrap aggregate**
  ``mid`` **F1** for rouge1 / rouge2 / rougeL, ``use_stemmer=False`` (HF default). Maps to Table columns **R-1**,
  **R-2**, **R-L**.
- **BERTScore**: ``evaluate``'s BERTScore — mean **F1** over sentences with ``lang='es'`` (Table **BERT**).

References are recovered by decoding the tokenized ``labels`` column. Use the same ``--tokenized-dir`` as training.

Example::

    python -m main.eval_decode --tokenized-dir data/processed/100k/tokenized --model-path runs/flan-t5-small-baseline/checkpoint-750 --tokenizer-model google/flan-t5-small --output-json runs/flan-t5-small-baseline/test_decode_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import evaluate
import torch
from datasets import Dataset, load_from_disk
from datasets.config import DATASETDICT_JSON_FILENAME
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from main.paths import resolve_under_repo


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Decode test split and compute BLEU, ROUGE, and BERTScore (paper Table 4 style).",
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
        help="Fine-tuned checkpoint dir (e.g. .../checkpoint-750 or .../final_model).",
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Paper Section 3: clinical reports generated with 512 output tokens.",
    )
    p.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="Paper Section 3: beam search 3.",
    )
    p.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=2,
        help="Paper Section 3: no repeating n-gram 2. Use 0 to disable.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Run model weights in float16 on CUDA (faster generation).",
    )
    p.add_argument(
        "--bleu-tokenize",
        type=str,
        default="13a",
        help=(
            "SacreBLEU tokenizer (default 13a = SacreBLEU / HF evaluate default for non-Chinese). "
            "Use intl for mteval-v14-style tokenization."
        ),
    )
    p.add_argument(
        "--bleu-lowercase",
        action="store_true",
        help="Lowercase before SacreBLEU (default: preserve case).",
    )
    p.add_argument(
        "--bertscore-lang",
        type=str,
        default="es",
        help="BERTScore language code (default es for Colombian Spanish).",
    )
    p.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=32,
        help="Batch size passed to evaluate bertscore.",
    )
    return p.parse_args(argv)


def _is_dataset_dict_dir(path: Path) -> bool:
    return path.is_dir() and (path / DATASETDICT_JSON_FILENAME).is_file()


def _resolve_tokenized_dataset_dir(path: Path) -> Path:
    if _is_dataset_dict_dir(path):
        return path
    nested = path / "tokenized"
    if _is_dataset_dict_dir(nested):
        return nested
    raise SystemExit(
        f"Not a Hugging Face DatasetDict directory: {path}\n"
        f"Expected {DATASETDICT_JSON_FILENAME} here or under {path / 'tokenized'}."
    )


def _log(msg: str) -> None:
    print(f"[eval_decode] {msg}", flush=True)


def _labels_to_text(label_ids: Sequence[int], tokenizer: Any) -> str:
    ids = [int(t) for t in label_ids if int(t) != -100]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _pick_device(cpu_only: bool) -> torch.device:
    if cpu_only:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_evaluate_metrics() -> tuple[Any, Any, Any]:
    """Load HF evaluate metrics once (SacreBLEU + ROUGE + BERTScore)."""
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    bertscore = evaluate.load("bertscore")
    return rouge, bleu, bertscore


def _subset_metrics_paper_table(
    hypotheses: list[str],
    references: list[str],
    *,
    rouge_m: Any,
    bleu_m: Any,
    bert_m: Any,
    bleu_tokenize: str,
    bleu_lowercase: bool,
    bertscore_lang: str,
    bertscore_batch_size: int,
    bert_device: str,
) -> dict[str, Any]:
    """Metrics aligned with paper Table 4 naming (R_1, R_2, R_L, BLEU, BERT)."""
    n = len(hypotheses)
    if n == 0:
        return {
            "n": 0,
            "R_1": 0.0,
            "R_2": 0.0,
            "R_L": 0.0,
            "BLEU": 0.0,
            "BERT": 0.0,
        }

    r_out = rouge_m.compute(
        predictions=hypotheses,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        use_aggregator=True,
    )
    b_out = bleu_m.compute(
        predictions=hypotheses,
        references=references,
        tokenize=bleu_tokenize,
        lowercase=bleu_lowercase,
        smooth_method="exp",
        use_effective_order=False,
    )
    bs_out = bert_m.compute(
        predictions=hypotheses,
        references=references,
        lang=bertscore_lang,
        batch_size=bertscore_batch_size,
        device=bert_device,
        rescale_with_baseline=False,
    )
    f1_list = bs_out["f1"]
    bert_mean = float(sum(f1_list) / len(f1_list)) if f1_list else 0.0

    return {
        "n": n,
        "R_1": float(r_out["rouge1"]),
        "R_2": float(r_out["rouge2"]),
        "R_L": float(r_out["rougeL"]),
        "BLEU": float(b_out["score"]) / 100.0,
        "BERT": bert_mean,
        "AVG": float(
            (float(r_out["rouge1"]) + float(r_out["rouge2"]) + float(r_out["rougeL"]) + float(b_out["score"]) / 100.0 + bert_mean)
            / 5.0
        ),
        "diagnostics": {
            "sacrebleu_score_0_100": float(b_out["score"]),
            "sacrebleu_bp": float(b_out["bp"]),
            "sacrebleu_precisions_0_100": [float(x) for x in b_out["precisions"]],
            "bertscore_f1_per_example": [float(x) for x in f1_list],
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    torch.manual_seed(args.seed)

    tokenized = _resolve_tokenized_dataset_dir(resolve_under_repo(args.tokenized_dir))
    model_path = resolve_under_repo(args.model_path)
    if not model_path.is_dir():
        raise SystemExit(f"model path not found: {model_path}")

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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else None
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)

    hypotheses: list[str] = []
    references: list[str] = []
    groups: list[str | None] = []

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
            features = [
                {"input_ids": batch_rows["input_ids"][i], "attention_mask": batch_rows["attention_mask"][i]}
                for i in range(len(batch_rows["input_ids"]))
            ]
            padded = collator(features)
            input_ids = padded["input_ids"].to(device)
            attention_mask = padded["attention_mask"].to(device)
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for i, text in enumerate(decoded):
                idx = start + i
                label_ids = test_ds[idx]["labels"]
                ref = _labels_to_text(label_ids, tokenizer)
                hypotheses.append(text.strip())
                references.append(ref)
                if "group" in test_ds.column_names:
                    g = test_ds[idx]["group"]
                    groups.append(None if g is None else str(g))
                else:
                    groups.append(None)

    bert_device = "cuda" if device.type == "cuda" else "cpu"

    rouge_m, bleu_m, bert_m = _load_evaluate_metrics()

    report: dict[str, Any] = {
        "paper": {
            "title": "Acoustic-Driven Generation of Pathological Speech Reports Using Large Language Models",
            "metrics_citation": "Ref. [40] Chauhan & Daniel, Neural Process. Lett. 55, 12663–12717 (2023) (MT metric survey)",
            "table_4_columns": "R-1, R-2, R-L, BLEU, BERT, AVG (same averaging as Table 4: mean of those five scores)",
            "generation_section_3": {
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
                "no_repeat_ngram_size": args.no_repeat_ngram_size if args.no_repeat_ngram_size > 0 else None,
            },
        },
        "tokenized_dir": str(tokenized),
        "model_path": str(model_path),
        "tokenizer_model": args.tokenizer_model,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "no_repeat_ngram_size": args.no_repeat_ngram_size if args.no_repeat_ngram_size > 0 else None,
            "batch_size": args.batch_size,
        },
        "metrics_config": {
            "stack": "huggingface evaluate (rouge, sacrebleu, bertscore)",
            "rouge": {
                "use_stemmer": False,
                "use_aggregator": True,
                "aggregate": "bootstrap mid F-measure (evaluate default)",
                "rouge_types": ["rouge1", "rouge2", "rougeL"],
            },
            "bleu": {
                "backend": "SacreBLEU via evaluate",
                "tokenize": args.bleu_tokenize,
                "smooth_method": "exp",
                "lowercase": args.bleu_lowercase,
                "use_effective_order": False,
                "reported_as": "BLEU in [0,1] = sacrebleu score / 100",
            },
            "bertscore": {
                "lang": args.bertscore_lang,
                "batch_size": args.bertscore_batch_size,
                "reported_as": "BERT = mean per-example F1 (evaluate bertscore)",
            },
        },
        "overall": _subset_metrics_paper_table(
            hypotheses,
            references,
            rouge_m=rouge_m,
            bleu_m=bleu_m,
            bert_m=bert_m,
            bleu_tokenize=args.bleu_tokenize,
            bleu_lowercase=args.bleu_lowercase,
            bertscore_lang=args.bertscore_lang,
            bertscore_batch_size=args.bertscore_batch_size,
            bert_device=bert_device,
        ),
    }

    if any(g is not None for g in groups):
        by_group: dict[str, dict[str, Any]] = {}
        for label in ("PD", "HC"):
            idxs = [i for i, g in enumerate(groups) if g == label]
            if not idxs:
                continue
            h = [hypotheses[i] for i in idxs]
            r = [references[i] for i in idxs]
            by_group[label] = _subset_metrics_paper_table(
                h,
                r,
                rouge_m=rouge_m,
                bleu_m=bleu_m,
                bert_m=bert_m,
                bleu_tokenize=args.bleu_tokenize,
                bleu_lowercase=args.bleu_lowercase,
                bertscore_lang=args.bertscore_lang,
                bertscore_batch_size=args.bertscore_batch_size,
                bert_device=bert_device,
            )
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
