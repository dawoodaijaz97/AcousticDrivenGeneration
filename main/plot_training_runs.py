from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from main.paths import repo_root, resolve_under_repo


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot train/eval loss curves from Hugging Face Trainer trainer_state.json "
            "so you can compare multiple fine-tuning runs. "
            "If <run_dir>/test_eval.json exists (from main.train post-train test), "
            "test_loss is drawn on the eval panel at the run's final step. "
            "If <run_dir>/test_decode_metrics.json exists (from main.eval_decode), "
            "overall R_1/R_2/R_L/BLEU/BERT/AVG are shown as grouped bars in a third panel."
        ),
    )
    p.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help=(
            "Training output directories (each may contain checkpoint-*/trainer_state.json). "
            "Repo-relative if not absolute."
        ),
    )
    p.add_argument(
        "--runs-parent",
        type=Path,
        default=None,
        help="If set, every immediate child directory is treated as a run_dir (skips missing trainer_state).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Figure path (.png / .pdf / .svg). Default: <repo>/runs/training_compare.png",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels (same count as runs after resolution). Default: run folder names.",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title (default: auto from run names).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Raster resolution when saving PNG.",
    )
    return p.parse_args(argv)


def _checkpoint_step(path: Path) -> int:
    name = path.name
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.split("-", 1)[1])
    except ValueError:
        return -1


def _dedupe_run_dirs(paths: list[Path]) -> list[Path]:
    """Same run folder can be added twice if --runs-parent and an explicit path overlap."""
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def find_trainer_state(run_dir: Path) -> Path | None:
    """Prefer the newest checkpoint's trainer_state.json (highest checkpoint-* step)."""
    direct = run_dir / "trainer_state.json"
    if direct.is_file():
        return direct

    checkpoints = [p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=_checkpoint_step, reverse=True)
    for cp in checkpoints:
        ts = cp / "trainer_state.json"
        if ts.is_file():
            return ts
    return None


def load_run_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_root_from_trainer_state(ts_path: Path) -> Path:
    """Training output dir: parent of checkpoint-* or the dir containing trainer_state.json."""
    if ts_path.parent.name.startswith("checkpoint-"):
        return ts_path.parent.parent
    return ts_path.parent


def load_test_eval_loss(run_root: Path) -> float | None:
    """Loss on real test split from main.train (test_eval.json); None if missing."""
    path = run_root / "test_eval.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("test_metrics") or {}
    raw = metrics.get("test_loss")
    if raw is None:
        return None
    return float(raw)


DECODE_BAR_KEYS: tuple[str, ...] = ("R_1", "R_2", "R_L", "BLEU", "BERT", "AVG")
DECODE_BAR_LABELS: tuple[str, ...] = ("R-1", "R-2", "R-L", "BLEU", "BERT", "AVG")


def load_test_decode_metrics(run_root: Path) -> dict[str, Any] | None:
    """Full JSON from main.eval_decode (test_decode_metrics.json); None if missing."""
    path = run_root / "test_decode_metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def overall_decode_bar_values(data: dict[str, Any]) -> list[float] | None:
    """Six overall scores for the decode bar panel, or None if not present."""
    overall = data.get("overall")
    if not isinstance(overall, dict):
        return None
    try:
        return [float(overall[k]) for k in DECODE_BAR_KEYS]
    except (KeyError, TypeError, ValueError):
        return None


def split_history(
    log_history: list[dict[str, Any]],
) -> tuple[
    list[tuple[int, float]],
    list[tuple[int, float]],
    list[tuple[int, float]],
    list[tuple[int, float]],
]:
    """Return (train_loss, eval_val_loss, eval_train_loss, lr) as (step, value) lists."""
    train: list[tuple[int, float]] = []
    eval_val: list[tuple[int, float]] = []
    eval_train: list[tuple[int, float]] = []
    lr: list[tuple[int, float]] = []
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        if "eval_val_loss" in row:
            eval_val.append((int(step), float(row["eval_val_loss"])))
        elif "eval_loss" in row:
            eval_val.append((int(step), float(row["eval_loss"])))
        if "eval_train_loss" in row:
            eval_train.append((int(step), float(row["eval_train_loss"])))
        if "loss" in row and not (
            "eval_loss" in row or "eval_val_loss" in row or "eval_train_loss" in row
        ):
            train.append((int(step), float(row["loss"])))
        if "learning_rate" in row:
            lr.append((int(step), float(row["learning_rate"])))
    train.sort(key=lambda x: x[0])
    eval_val.sort(key=lambda x: x[0])
    eval_train.sort(key=lambda x: x[0])
    lr.sort(key=lambda x: x[0])
    return train, eval_val, eval_train, lr


def _try_style() -> None:
    import matplotlib.pyplot as plt

    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def plot_comparison(
    series: list[tuple[str, Path, dict[str, Any]]],
    *,
    title: str | None,
    output: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    _try_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=False, height_ratios=[2.2, 1.0, 1.15])
    ax_train, ax_eval, ax_decode = axes[0], axes[1], axes[2]

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    summary_lines: list[str] = []

    decode_rows: list[tuple[int, str, list[float]]] = []

    for i, (label, ts_path, state) in enumerate(series):
        color = colors[i % len(colors)]
        train, eval_val, eval_train, _lr = split_history(state.get("log_history", []))
        ax_train.plot([s for s, _ in train], [y for _, y in train], "-", color=color, label=label, alpha=0.9, lw=1.8)
        if eval_val:
            ax_eval.plot(
                [s for s, _ in eval_val],
                [y for _, y in eval_val],
                "o-",
                color=color,
                label=f"{label} val",
                alpha=0.9,
                lw=1.5,
                ms=5,
            )
        if eval_train:
            ax_eval.plot(
                [s for s, _ in eval_train],
                [y for _, y in eval_train],
                "s--",
                color=color,
                label=f"{label} train",
                alpha=0.85,
                lw=1.4,
                ms=4,
            )

        run_root = run_root_from_trainer_state(ts_path)
        test_loss = load_test_eval_loss(run_root)
        final_step = state.get("global_step")
        if test_loss is not None and isinstance(final_step, int):
            ax_eval.scatter(
                [final_step],
                [test_loss],
                marker="*",
                s=140,
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=6,
                label=f"{label} test (real)",
            )
        elif test_loss is not None:
            ax_eval.axhline(test_loss, color=color, linestyle=":", alpha=0.85, lw=1.5, label=f"{label} test (real)")

        decode_json = load_test_decode_metrics(run_root)
        decode_vals = overall_decode_bar_values(decode_json) if decode_json else None
        if decode_vals is not None:
            decode_rows.append((i, label, decode_vals))

        best = state.get("best_metric")
        step = state.get("global_step")
        summary_lines.append(
            f"{label}: best_eval_loss={best:.6f}" if isinstance(best, (int, float)) else f"{label}: (no best_metric)"
        )
        summary_lines[-1] += f" @ step {step}" if step is not None else ""
        summary_lines[-1] += f" — {ts_path.parent.name}/{ts_path.name}"
        if test_loss is not None:
            summary_lines.append(f"  └ test_loss (real)={test_loss:.6f}  <- {run_root / 'test_eval.json'}")
        else:
            summary_lines.append(f"  └ no test_eval.json under {run_root}")
        if decode_vals is not None:
            summary_lines.append(
                f"  └ decode overall: "
                + ", ".join(f"{k}={v:.3f}" for k, v in zip(DECODE_BAR_KEYS, decode_vals))
                + f"  <- {run_root / 'test_decode_metrics.json'}"
            )
            if isinstance(decode_json, dict):
                bg = decode_json.get("by_group")
                if isinstance(bg, dict):
                    for gname in ("PD", "HC"):
                        block = bg.get(gname)
                        if not isinstance(block, dict):
                            continue
                        try:
                            parts = [f"{k}={float(block[k]):.3f}" for k in DECODE_BAR_KEYS if k in block]
                            if parts:
                                summary_lines.append(f"     {gname}: " + ", ".join(parts))
                        except (TypeError, ValueError):
                            pass
        else:
            summary_lines.append(f"  └ no test_decode_metrics.json under {run_root}")

    ax_train.set_ylabel("train loss")
    ax_train.set_title(title or "Training run comparison")
    ax_train.legend(loc="upper right", fontsize=9)
    ax_train.grid(True, alpha=0.35)

    ax_eval.set_ylabel("eval / test loss (val o / train sq / test * real)")
    ax_eval.set_xlabel("step")
    ax_eval.legend(loc="upper right", fontsize=9)
    ax_eval.grid(True, alpha=0.35)

    n_decode = len(decode_rows)
    x = np.arange(len(DECODE_BAR_LABELS))
    if n_decode == 0:
        ax_decode.set_title("Decode metrics (test_decode_metrics.json — none found)")
        ax_decode.text(0.5, 0.5, "No test_decode_metrics.json in run dirs", ha="center", va="center", transform=ax_decode.transAxes)
        ax_decode.set_xticks(x)
        ax_decode.set_xticklabels(DECODE_BAR_LABELS)
    else:
        ax_decode.set_title("Decode metrics — overall (main.eval_decode, Table 4 style)")
        width = 0.82 / n_decode
        centers = np.linspace(-(n_decode - 1) / 2, (n_decode - 1) / 2, n_decode) * width * 1.15
        for j, (run_i, lab, vals) in enumerate(decode_rows):
            color = colors[run_i % len(colors)]
            offset = centers[j]
            ax_decode.bar(x + offset, vals, width=width * 0.92, label=lab, color=color, alpha=0.88, edgecolor="black", linewidth=0.35)
        ax_decode.set_xticks(x)
        ax_decode.set_xticklabels(DECODE_BAR_LABELS)
        ax_decode.set_ylim(0.0, 1.05)
        ax_decode.legend(loc="upper right", fontsize=8, ncol=min(3, n_decode))
    ax_decode.set_ylabel("score (0–1)")
    ax_decode.grid(True, axis="y", alpha=0.35)

    fig.text(0.5, 0.01, "\n".join(summary_lines), ha="center", va="bottom", fontsize=8, family="monospace")
    fig.subplots_adjust(bottom=0.26, top=0.92, hspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = repo_root()

    run_paths: list[Path] = []
    if args.runs_parent is not None:
        parent = resolve_under_repo(args.runs_parent)
        if not parent.is_dir():
            raise SystemExit(f"runs-parent not a directory: {parent}")
        for child in sorted(parent.iterdir()):
            if child.is_dir() and find_trainer_state(child):
                run_paths.append(child)
    run_paths.extend(resolve_under_repo(p) for p in args.run_dirs)
    n_before = len(run_paths)
    run_paths = _dedupe_run_dirs(run_paths)
    if len(run_paths) < n_before:
        print(
            f"[plot_training_runs] note: removed {n_before - len(run_paths)} duplicate run path(s) "
            "(same folder listed more than once, e.g. --runs-parent plus an explicit run dir).",
            file=sys.stderr,
        )

    if not run_paths:
        raise SystemExit("No run directories: pass run_dirs and/or --runs-parent with subfolders containing trainer_state.")

    staged: list[tuple[Path, Path, dict[str, Any]]] = []
    for rd in run_paths:
        ts = find_trainer_state(rd)
        if ts is None:
            print(f"[plot_training_runs] skip (no trainer_state): {rd}", file=sys.stderr)
            continue
        staged.append((rd, ts, load_run_state(ts)))

    if not staged:
        raise SystemExit("No trainer_state.json found under the given run directories.")

    labels = list(args.labels) if args.labels else []
    if labels and len(labels) != len(staged):
        print(
            f"[plot_training_runs] warning: got {len(labels)} --labels but {len(staged)} runs; "
            "using folder names for unmatched entries.",
            file=sys.stderr,
        )
    resolved: list[tuple[str, Path, dict[str, Any]]] = []
    for i, (rd, ts, state) in enumerate(staged):
        label = labels[i] if i < len(labels) else rd.name
        resolved.append((label, ts, state))

    out = args.output
    if out is None:
        out = root / "runs" / "training_compare.png"
    else:
        out = resolve_under_repo(out)

    plot_comparison(resolved, title=args.title, output=out, dpi=args.dpi)
    print(f"[plot_training_runs] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
