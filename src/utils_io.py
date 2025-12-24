"""Utility helpers for safe IO operations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import json

import pandas as pd

from . import config

PathLike = Union[str, Path]


def resolve_path(path: PathLike, base_dir: Path | None = None) -> Path:
    """Resolve *path* within *base_dir* (defaults to project root)."""
    base = base_dir or config.PROJECT_ROOT
    resolved = (base / path).resolve() if not str(path).startswith("/") else Path(path).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise ValueError(f"Path {resolved} escapes base directory {base}")
    return resolved


def ensure_directory(path: PathLike) -> Path:
    """Create parent directories for *path* and return the Path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def load_csv(path: PathLike) -> pd.DataFrame:
    """Load a CSV file, ensuring it is non-empty."""
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing file: {resolved}")
    df = pd.read_csv(resolved)
    if df.empty:
        raise ValueError(f"File {resolved} is empty")
    return df


def save_csv(df: pd.DataFrame, path: PathLike, index: bool = False) -> None:
    """Persist a dataframe as CSV."""
    target = ensure_directory(path)
    df.to_csv(target, index=index)


def save_json(data: Any, path: PathLike, indent: int = 2) -> None:
    """Persist JSON data to disk."""
    target = ensure_directory(path)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def write_text(text: str, path: PathLike) -> None:
    """Write text to file, creating directories when needed."""
    target = ensure_directory(path)
    target.write_text(text, encoding="utf-8")
