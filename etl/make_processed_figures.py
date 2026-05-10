from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURE_COLS = [
    "breathing",
    "lips",
    "palate",
    "larynx",
    "monotonicity",
    "tongue",
    "intelligibility",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    p_parquet = processed_dir / f"{split}.parquet"
    p_csv = processed_dir / f"{split}.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    return pd.read_csv(p_csv)


def write_figures(processed_dir: Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    out_dir = processed_dir / "dashboard_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        df = load_split(processed_dir, split)

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        for i, col in enumerate(FEATURE_COLS):
            ax = axes[i]
            ax.hist(df[col].dropna().values, bins=40)
            ax.set_title(col)
        for j in range(len(FEATURE_COLS), len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{processed_dir.name} • {split}: feature histograms", y=0.98)
        fig.tight_layout()
        fig.savefig(out_dir / f"{split}_features.png", dpi=160)
        plt.close(fig)

        if "input_text" in df.columns and "target_text" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(df["input_text"].astype(str).str.len().values, bins=60)
            axes[0].set_title("input_text length (chars)")
            axes[1].hist(df["target_text"].astype(str).str.len().values, bins=60)
            axes[1].set_title("target_text length (chars)")
            fig.suptitle(f"{processed_dir.name} • {split}: text length histograms", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / f"{split}_lengths.png", dpi=160)
            plt.close(fig)

    print(f"[OK] wrote figures to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", default="all", choices=["all", "1k", "10k", "100k"])
    args = ap.parse_args()

    root = repo_root()
    processed_root = root / "processed"
    sizes = ["1k", "10k", "100k"] if args.train_size == "all" else [args.train_size]

    for s in sizes:
        write_figures(processed_root / s)


if __name__ == "__main__":
    main()

