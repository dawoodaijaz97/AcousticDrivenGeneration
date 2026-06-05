from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from datasets.config import DATASETDICT_JSON_FILENAME
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from main.paths import repo_root, resolve_under_repo


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a T5/Flan-T5 seq2seq model on tokenized acoustic→report data.",
    )
    p.add_argument(
        "--tokenized-dir",
        type=Path,
        required=True,
        help=(
            "DatasetDict from prepare --tokenize (contains dataset_dict.json), or a prepare output dir "
            "that has a tokenized/ subfolder. Repo-relative; prefer forward slashes in Git Bash."
        ),
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-small",
        help="HF model id (should match the tokenizer used when building --tokenized-dir).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Checkpoints and final model export. Repo-relative if not absolute; prefer forward slashes in Git Bash.",
    )
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="If >0, overrides num_train_epochs.")
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--per-device-eval-batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for seq2seq cross-entropy (0.0 disables smoothing).",
    )
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help=(
            "Checkpoint interval. With load_best_model_at_end=True (always on here), must be a positive "
            "multiple of --eval-steps (e.g. eval every 7000 → save every 7000)."
        ),
    )
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Resume training from a checkpoint. Use 'auto' (or 'true') to resume from the latest "
            "checkpoint-* in --output-dir, or pass an explicit checkpoint path. Omit to train from scratch. "
            "Requires checkpoints saved by a prior run (optimizer/scheduler/RNG state included)."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training rows for debugging.",
    )
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on validation rows for debugging.",
    )
    p.add_argument(
        "--eval-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also run eval on the train split each eval step (logs eval_train_* metrics). "
            "Disable with --no-eval-train to save time."
        ),
    )
    p.add_argument(
        "--eval-train-max-samples",
        type=int,
        default=5000,
        help=(
            "Max train rows used for train-split eval (default caps cost on large splits). "
            "Use 0 for the full train set."
        ),
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 (CUDA only; ignored on CPU).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 when supported (CUDA / some CPUs).",
    )
    p.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=("none", "wandb", "tensorboard"),
    )
    p.add_argument(
        "--predict-with-generate",
        action="store_true",
        help="Run generation during eval (slower; no extra metrics unless you add compute_metrics).",
    )
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit if no CUDA GPU is visible to PyTorch (catches CPU-only PyTorch or driver issues).",
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU training (overrides automatic CUDA/MPS selection).",
    )
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on test rows for post-train evaluation (real reports split).",
    )
    p.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="If the tokenized DatasetDict has a test split, skip final evaluate() on it.",
    )
    p.add_argument(
        "--lm-studio-base-url",
        type=str,
        default=None,
        help=(
            "Optional: write lm_studio.json in --output-dir for eval hints. "
            "Training always loads --model-name from Hugging Face locally (LM Studio does not train over HTTP)."
        ),
    )
    p.add_argument(
        "--lm-studio-model",
        type=str,
        default=None,
        help="Optional: LM Studio model id included in lm_studio.json.",
    )
    return p.parse_args(argv)


def _maybe_subset(ds: Dataset, n: int | None) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.select(range(n))


def _metrics_for_json(metrics: dict) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {}
    for k, v in metrics.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (bool, str)):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = v
        elif hasattr(v, "item"):
            out[k] = float(v.item())
        else:
            out[k] = float(v)
    return out


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def _mps_is_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _resolve_use_cpu(args: argparse.Namespace) -> bool:
    if args.cpu_only:
        return True
    if torch.cuda.is_available() or _mps_is_available():
        return False
    return True


def _log_device_plan(use_cpu: bool) -> None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        _log(f"CUDA_VISIBLE_DEVICES = {vis!r}")
    if _env_flag("ACCELERATE_USE_CPU") and torch.cuda.is_available() and not use_cpu:
        _log(
            "WARNING: ACCELERATE_USE_CPU is set; Accelerate may still force CPU. "
            "Unset it for GPU training."
        )
    if use_cpu:
        _log("device            = CPU (use_cpu=True)")
        if torch.cuda.is_available():
            _log("WARNING: CUDA is available but --cpu-only was set.")
        return
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        _log(f"device            = CUDA ({name}, device_count={torch.cuda.device_count()})")
        return
    if _mps_is_available():
        _log("device            = MPS (Apple Metal)")
        return
    _log("device            = CPU (no CUDA or MPS in this PyTorch build)")


def _is_dataset_dict_dir(path: Path) -> bool:
    return path.is_dir() and (path / DATASETDICT_JSON_FILENAME).is_file()


def _resolve_tokenized_dataset_dir(path: Path) -> Path:
    """Accept either the HF folder (with dataset_dict.json) or a prepare output dir containing tokenized/."""
    if _is_dataset_dict_dir(path):
        return path
    nested = path / "tokenized"
    if _is_dataset_dict_dir(nested):
        _log(f"using nested DatasetDict -> {nested}")
        return nested
    hint = ""
    if path.is_dir() and (path / "train.jsonl").is_file():
        hint = (
            "\nThis folder looks like prepare output without --tokenize (e.g. train.jsonl). "
            "Re-run prepare with --tokenize; tokenized data is written to <output-dir>/tokenized/ by default."
        )
    raise SystemExit(
        f"Not a Hugging Face DatasetDict directory: {path}\n"
        f"Expected {DATASETDICT_JSON_FILENAME} here or under {path / 'tokenized'}.{hint}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    tokenized_arg = resolve_under_repo(args.tokenized_dir)
    out = resolve_under_repo(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not tokenized_arg.is_dir():
        raise SystemExit(f"tokenized dir not found: {tokenized_arg}")

    _log(f"repo_root         = {repo_root()}")
    tokenized = _resolve_tokenized_dataset_dir(tokenized_arg)
    _log(f"tokenized_dir     = {tokenized}")
    _log(f"output_dir        = {out}")
    _log(f"model_name        = {args.model_name}")
    if args.lm_studio_base_url:
        _log(
            "lm_studio         = --lm-studio-base-url set (hint file only; "
            "weights still loaded from --model-name for training)"
        )

    raw = load_from_disk(str(tokenized))
    if "train" not in raw or "val" not in raw:
        raise SystemExit("DatasetDict must contain 'train' and 'val' splits.")

    train_ds = _maybe_subset(raw["train"], args.max_train_samples)
    eval_ds = _maybe_subset(raw["val"], args.max_eval_samples)
    test_ds: Dataset | None = None
    if "test" in raw:
        test_ds = _maybe_subset(raw["test"], args.max_test_samples)
    _log(f"train rows        = {len(train_ds):,}")
    _log(f"val rows          = {len(eval_ds):,}")
    if test_ds is not None:
        _log(f"test rows         = {len(test_ds):,} (post-train eval if not --skip-test-eval)")
    else:
        _log("test split        = (not in DatasetDict; no post-train test eval)")

    eval_train_ds: Dataset | None = None
    if args.eval_train:
        cap = args.eval_train_max_samples
        if cap == 0:
            eval_train_ds = train_ds
            _log(f"train eval rows   = {len(eval_train_ds):,} (full train set)")
        else:
            eval_train_ds = _maybe_subset(train_ds, cap)
            _log(f"train eval rows   = {len(eval_train_ds):,} (cap={cap:,})")

    if args.require_gpu and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to PyTorch (torch.cuda.is_available() is False). "
            "Install a CUDA-enabled build from https://pytorch.org or conda, and verify drivers. "
            "Omit --require-gpu if you intend to train on CPU."
        )
    if args.cpu_only and args.require_gpu:
        raise SystemExit("Choose at most one of --cpu-only and --require-gpu.")
    if args.eval_train_max_samples < 0:
        raise SystemExit("--eval-train-max-samples must be >= 0 (0 = use full train set for train eval).")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise SystemExit("--label-smoothing must be in [0.0, 1.0).")

    use_cpu = _resolve_use_cpu(args)
    _log_device_plan(use_cpu)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    use_fp16 = bool(args.fp16 and torch.cuda.is_available() and not use_cpu)
    use_bf16 = bool(args.bf16 and not use_cpu and (torch.cuda.is_available() or _mps_is_available()))

    eval_steps = args.eval_steps
    save_steps = args.save_steps
    if args.max_steps > 0:
        eval_steps = max(1, min(eval_steps, args.max_steps))
        save_steps = max(1, min(save_steps, args.max_steps))

    # HF Trainer + load_best_model_at_end: save_steps must be a positive multiple of eval_steps.
    if save_steps % eval_steps != 0:
        k = max(1, (save_steps + eval_steps - 1) // eval_steps)
        suggestion = k * eval_steps
        raise SystemExit(
            "With load_best_model_at_end=True, --save-steps must be a positive integer multiple of "
            f"--eval-steps (Hugging Face requirement). Got save_steps={save_steps}, eval_steps={eval_steps}. "
            f"Example: --save-steps {eval_steps}  or  --save-steps {suggestion}"
        )

    metric_for_best = "eval_val_loss" if eval_train_ds is not None else "eval_loss"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        use_cpu=use_cpu,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none" if args.report_to == "none" else args.report_to,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=512,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )

    eval_payload: Dataset | dict[str, Dataset]
    if eval_train_ds is not None:
        eval_payload = {"val": eval_ds, "train": eval_train_ds}
    else:
        eval_payload = eval_ds

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_payload,
        processing_class=tokenizer,
        data_collator=collator,
    )

    _log(f"Trainer.device    = {training_args.device}")

    resume_arg: bool | str | None
    rfc = args.resume_from_checkpoint
    if rfc is None or rfc == "":
        resume_arg = None
    elif rfc.strip().lower() in ("auto", "true", "1", "yes", "latest"):
        resume_arg = True
        _log("resume            = latest checkpoint in --output-dir")
    else:
        ckpt = resolve_under_repo(Path(rfc))
        if not ckpt.is_dir():
            raise SystemExit(f"--resume-from-checkpoint path not found: {ckpt}")
        resume_arg = str(ckpt)
        _log(f"resume            = {ckpt}")

    trainer.train(resume_from_checkpoint=resume_arg)

    if test_ds is not None and not args.skip_test_eval:
        _log("test evaluation   = running on best loaded model (test split, real reports)")
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
        report = {
            "test_metrics": _metrics_for_json(test_metrics),
            "test_rows": len(test_ds),
            "tokenized_dir": str(tokenized),
            "output_dir": str(out),
            "model_name": args.model_name,
        }
        report_path = out / "test_eval.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _log(f"test_eval.json    = {report_path}")
        for key in sorted(test_metrics.keys()):
            _log(f"  {key}: {test_metrics[key]}")

    trainer.save_model(str(out / "final_model"))
    tokenizer.save_pretrained(str(out / "final_model"))
    if args.lm_studio_base_url:
        hint = {
            "base_url": args.lm_studio_base_url.strip(),
            "model": args.lm_studio_model,
            "tokenizer_model": args.model_name,
            "note": "Training used local Hugging Face weights. For decode via LM Studio, use main.eval_decode --backend lm-studio.",
        }
        hint_path = out / "lm_studio.json"
        hint_path.write_text(json.dumps(hint, indent=2), encoding="utf-8")
        _log(f"wrote lm_studio -> {hint_path}")

    _log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
