from __future__ import annotations

import re
from pathlib import Path

# Git Bash / MSYS often strip ``\`` in unquoted args, so ``data\processed\1k`` becomes ``dataprocessed1k``.
_COLLAPSED_DATAPROCESSED = re.compile(r"(?i)^dataprocessed(100k|10k|1k)$")


def repo_root() -> Path:
    """Repository root (parent of the ``main`` package)."""
    return Path(__file__).resolve().parent.parent


def data_splits_dir(root: Path | None = None) -> Path:
    base = root if root is not None else repo_root()
    return base / "data" / "Data_splits"


def repair_shell_collapsed_path(path: Path | str) -> Path:
    """Expand a single-segment typo produced when a POSIX shell eats ``\\`` in ``data\\processed\\1k``."""
    p = Path(path)
    if p.is_absolute() or len(p.parts) != 1:
        return p
    m = _COLLAPSED_DATAPROCESSED.fullmatch(p.parts[0])
    if not m:
        return p
    size = m.group(1).lower()
    return Path("data") / "processed" / size


def resolve_under_repo(path: Path | str, root: Path | None = None) -> Path:
    """Return an absolute path. If ``path`` is relative, resolve it under ``root`` (default: repo root), not cwd."""
    p = repair_shell_collapsed_path(path)
    base = root if root is not None else repo_root()
    return (base / p).resolve() if not p.is_absolute() else p.resolve()
