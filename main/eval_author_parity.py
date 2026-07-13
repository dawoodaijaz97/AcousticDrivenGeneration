"""Author-parity decode eval — score a fine-tuned checkpoint with the *original
authors'* metric definitions and generation settings, not this repo's honest ones.

The paper's public code (``Speech_LLM/4_model_evaluation_flan_t5.py``) reports
Flan-T5 quality with three choices that differ from :mod:`main.eval_decode`:

1. **Character-level BLEU-2, not word-level BLEU-4.** Line ~113 passes raw
   *strings* to ``nltk.translate.bleu_score.sentence_bleu([ref], hyp,
   weights=(0.5, 0.5))``. nltk expects token lists; given a string it iterates
   over **characters**, so this is char-level unigram+bigram BLEU. Two English
   clinical reports share an alphabet and heavy boilerplate, so this scores far
   higher (~0.8) than word BLEU-4 (~0.4) for the *same* text. ``eval_decode``
   uses SacreBLEU corpus BLEU-4 (13a) — the standard, stricter metric.
2. **Stochastic decode with a length floor.** They generate with
   ``do_sample=True, temperature=0.7, top_k=50, num_beams=3,
   no_repeat_ngram_size=2, min_length=100, max_length=512`` (``eval_decode``
   uses deterministic beam search and no ``min_length``).
3. **BERTScore ``lang="en"``.** The reports are English; ``eval_decode``
   defaults to ``es`` (a real bug there). This script uses ``en`` to match.

This is **not** a replacement for :mod:`main.eval_decode` — decode AVG for model
*selection* must still come from that script's honest metrics. This module exists
only to answer one question: *under the original authors' ruler, does our best
checkpoint match or beat their reported Flan-T5 row?* It reuses the tokenized
``DatasetDict`` (so the encoder sees exactly the input the model was trained on)
and only swaps the generation kwargs and the metric implementations.

ROUGE here uses ``rouge_score.RougeScorer(..., use_stemmer=True)`` and per-example
mean F-measure (the authors' aggregation), unlike ``eval_decode``'s bootstrap-mid
``evaluate`` ROUGE with ``use_stemmer=False``. BLEU/ROUGE/BERT are all averaged
**per example**, matching the authors' loop, not corpus-aggregated.

Example (HPC, on the B17 merged checkpoint)::

    python -m main.eval_author_parity \
      --tokenized-dir data/processed/flan-t5-base/100k-flan-paper/tokenized \
      --model-path runs/flan-t5-base/100k-flan-paper-5ep-lora32/final_model \
      --tokenizer-model google/flan-t5-base \
      --output-json runs/flan-t5-base/100k-flan-paper-5ep-lora32/test_author_parity_metrics.json \
      --require-gpu
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import evaluate
import nltk
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from datasets.config import DATASETDICT_JSON_FILENAME
from rouge_score import rouge_scorer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from main.paths import resolve_under_repo


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Decode the test split and score with the ORIGINAL authors' metrics "
            "(char-level BLEU-2, rouge_score stemmer F1, BERTScore lang=en) and "
            "generation settings (sampling + min_length). For parity checks only; "
            "use main.eval_decode for model selection."
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
        help="Fine-tuned checkpoint dir (LoRA runs: the merged final_model/).",
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
    p.add_argument("--batch-size", type=int, default=8)

    # --- Original authors' generation settings (Speech_LLM/4_model_evaluation_flan_t5.py) ---
    p.add_argument("--num-beams", type=int, default=3, help="Authors: beam search 3.")
    p.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=2,
        help="Authors: no repeating n-gram 2. Use 0 to disable.",
    )
    p.add_argument(
        "--min-length",
        type=int,
        default=100,
        help="Authors: min_length=100 (forces long outputs).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Authors: max_length=512 (total generated length, not max_new_tokens).",
    )
    p.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        default=True,
        help="Authors: do_sample=True (default here).",
    )
    p.add_argument(
        "--no-sample",
        dest="do_sample",
        action="store_false",
        help="Deterministic decode instead of the authors' sampling.",
    )
    p.add_argument("--temperature", type=float, default=0.7, help="Authors: temperature=0.7.")
    p.add_argument("--top-k", type=int, default=50, help="Authors: top_k=50.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--cpu-only", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit if no CUDA GPU is visible (sampling decode on CPU is very slow).",
    )
    p.add_argument("--fp16", action="store_true", help="Run model weights in float16 on CUDA.")
    p.add_argument(
        "--bertscore-lang",
        type=str,
        default="en",
        help="BERTScore language code (default en: the reports are English).",
    )
    p.add_argument("--bertscore-batch-size", type=int, default=32)
    return p.parse_args(argv)


def _log(msg: str) -> None:
    print(f"[eval_author_parity] {msg}", flush=True)


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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _char_bleu2(reference: str, hypothesis: str) -> float:
    """Reproduce the authors' exact call: raw strings to nltk sentence_bleu with
    weights (0.5, 0.5). nltk iterates strings as characters, so this is char-level
    unigram+bigram BLEU. No smoothing function, matching the original code."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(
            nltk.translate.bleu_score.sentence_bleu(
                [reference], hypothesis, weights=(0.5, 0.5)
            )
        )


def _subset_metrics_author(
    hypotheses: list[str],
    references: list[str],
    *,
    scorer: rouge_scorer.RougeScorer,
    bert_m: Any,
    bertscore_lang: str,
    bertscore_batch_size: int,
    bert_device: str,
) -> dict[str, Any]:
    """Per-example mean metrics with the authors' definitions (char BLEU-2,
    stemmed ROUGE F1, BERTScore-en F1)."""
    n = len(hypotheses)
    if n == 0:
        return {"n": 0, "R_1": 0.0, "R_2": 0.0, "R_L": 0.0, "BLEU": 0.0, "BERT": 0.0, "AVG": 0.0}

    r1, r2, rl, bleu = [], [], [], []
    for ref, hyp in zip(references, hypotheses):
        # rouge_score F-measure is symmetric in (target, prediction); order is irrelevant.
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
        bleu.append(_char_bleu2(ref, hyp))

    bs_out = bert_m.compute(
        predictions=hypotheses,
        references=references,
        lang=bertscore_lang,
        batch_size=bertscore_batch_size,
        device=bert_device,
        rescale_with_baseline=False,
    )
    bert_f1 = [float(x) for x in bs_out["f1"]]

    r1_m = float(np.mean(r1))
    r2_m = float(np.mean(r2))
    rl_m = float(np.mean(rl))
    bleu_m = float(np.mean(bleu))
    bert_m_ = float(np.mean(bert_f1)) if bert_f1 else 0.0

    return {
        "n": n,
        "R_1": r1_m,
        "R_2": r2_m,
        "R_L": rl_m,
        "BLEU": bleu_m,
        "BERT": bert_m_,
        "AVG": float((r1_m + r2_m + rl_m + bleu_m + bert_m_) / 5.0),
        "diagnostics": {
            "bleu_is": "char-level BLEU-2 (nltk raw-string call, weights=(0.5,0.5))",
            "bertscore_f1_per_example": bert_f1,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _seed_everything(args.seed)

    model_path = resolve_under_repo(args.model_path)
    if not model_path.is_dir():
        raise SystemExit(f"model path not found: {model_path}")

    if args.cpu_only and args.require_gpu:
        raise SystemExit("Choose at most one of --cpu-only and --require-gpu.")
    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch (torch.cuda.is_available() is False). "
            "Sampling decode on CPU is far too slow. Omit --require-gpu only if you "
            "intend to decode on CPU."
        )

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
    _log(f"tokenized_dir = {tokenized}")
    _log(f"model_path    = {model_path}")
    _log(f"device        = {device}")
    _log(f"test rows     = {len(test_ds):,}")
    _log(
        "decode = do_sample=%s temp=%s top_k=%s beams=%s min_len=%s max_len=%s no_repeat=%s"
        % (
            args.do_sample,
            args.temperature,
            args.top_k,
            args.num_beams,
            args.min_length,
            args.max_length,
            args.no_repeat_ngram_size,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else None
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), torch_dtype=dtype)
    model.eval()
    model.to(device)

    gen_kwargs: dict[str, Any] = {
        "max_length": args.max_length,
        "min_length": args.min_length,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "temperature": args.temperature,
        # Authors set pad_token_id=eos_token_id; replicate for faithfulness.
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    hypotheses: list[str] = []
    references: list[str] = []
    groups: list[str | None] = []
    sample_ids: list[str | None] = []

    bs = max(1, args.batch_size)
    n = len(test_ds)
    with torch.no_grad():
        for start in range(0, n, bs):
            batch_rows = test_ds[start : start + bs]
            batch_len = len(batch_rows["input_ids"])
            features = [
                {
                    "input_ids": batch_rows["input_ids"][i],
                    "attention_mask": batch_rows["attention_mask"][i],
                }
                for i in range(batch_len)
            ]
            padded = collator(features)
            input_ids = padded["input_ids"].to(device)
            attention_mask = padded["attention_mask"].to(device)
            gen = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for i, text in enumerate(decoded):
                idx = start + i
                ref = _labels_to_text(test_ds[idx]["labels"], tokenizer)
                hypotheses.append(text.strip())
                references.append(ref)
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
        pred_rows = [
            {
                "sample_id": sample_ids[i] if i < len(sample_ids) else None,
                "group": groups[i] if i < len(groups) else None,
                "reference": references[i],
                "hypothesis": hypotheses[i],
            }
            for i in range(len(hypotheses))
        ]
        pred_path = resolve_under_repo(args.output_predictions_json)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(
            json.dumps(
                {"model_path": str(model_path), "tokenized_dir": str(tokenized), "n": len(pred_rows), "rows": pred_rows},
                indent=2,
            ),
            encoding="utf-8",
        )
        _log(f"wrote predictions {pred_path}")

    bert_device = "cuda" if device.type == "cuda" else "cpu"
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bert_m = evaluate.load("bertscore")

    def _score(h: list[str], r: list[str]) -> dict[str, Any]:
        return _subset_metrics_author(
            h,
            r,
            scorer=scorer,
            bert_m=bert_m,
            bertscore_lang=args.bertscore_lang,
            bertscore_batch_size=args.bertscore_batch_size,
            bert_device=bert_device,
        )

    report: dict[str, Any] = {
        "note": (
            "AUTHOR-PARITY metrics — NOT for model selection. BLEU is char-level BLEU-2 "
            "(nltk raw-string bug reproduced), ROUGE is stemmed per-example mean F1, "
            "BERTScore lang=en. Compare against the paper's reported Flan-T5 numbers, "
            "not against main.eval_decode AVG (word SacreBLEU-4)."
        ),
        "tokenized_dir": str(tokenized),
        "model_path": str(model_path),
        "tokenizer_model": args.tokenizer_model,
        "generation": {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "num_beams": args.num_beams,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "no_repeat_ngram_size": args.no_repeat_ngram_size if args.no_repeat_ngram_size > 0 else None,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "metrics_config": {
            "bleu": "char-level BLEU-2 via nltk.sentence_bleu([ref], hyp, weights=(0.5,0.5)); per-example mean",
            "rouge": "rouge_score RougeScorer use_stemmer=True; per-example mean F-measure",
            "bertscore": {"lang": args.bertscore_lang, "reported_as": "mean per-example F1"},
        },
        "overall": _score(hypotheses, references),
    }

    if any(g is not None for g in groups):
        by_group: dict[str, dict[str, Any]] = {}
        for label in ("PD", "HC"):
            idxs = [i for i, g in enumerate(groups) if g == label]
            if not idxs:
                continue
            by_group[label] = _score([hypotheses[i] for i in idxs], [references[i] for i in idxs])
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
