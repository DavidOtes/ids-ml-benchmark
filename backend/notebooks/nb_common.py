"""Shared setup for the ft40 notebook study (top-40 features, multi-split).

Every notebook starts by importing this module. It resolves the per-dataset
artifact tree under artifacts/ft40/<dataset>/, configures logging to both the
notebook output and a per-notebook log file, and provides small helpers so
each step's facts land in saved artifacts (the dissertation's source of truth).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from ids import config  # noqa: E402


@dataclass(frozen=True)
class FtPaths:
    root: Path
    logs: Path
    eda: Path
    data: Path
    features: Path
    splits: Path
    runs: Path
    results: Path


def ft_paths(dataset: str, ft_root: str | Path | None = None) -> FtPaths:
    """Resolve (and create) the ft40 artifact subtree for one dataset."""
    if dataset not in config.VALID_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {config.VALID_DATASETS}")
    if ft_root is None:
        base = config.ARTIFACTS_DIR / "ft40"
    elif Path(ft_root).is_absolute():
        base = Path(ft_root)
    else:
        # relative names ("ft40_mi") resolve under the artifacts dir
        base = config.ARTIFACTS_DIR / ft_root
    root = base / dataset
    p = FtPaths(
        root=root, logs=root / "logs", eda=root / "eda", data=root / "data",
        features=root / "features", splits=root / "splits", runs=root / "runs",
        results=root / "results",
    )
    for d in vars(p).values():
        d.mkdir(parents=True, exist_ok=True)
    (p.eda / "plots").mkdir(exist_ok=True)
    (p.results / "plots").mkdir(exist_ok=True)
    return p


def setup_logging(log_file: Path, name: str) -> logging.Logger:
    """Log to the notebook output AND a persistent file, with timestamps.

    Re-running the setup cell replaces old handlers instead of stacking them.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger(name)


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


@contextmanager
def timed(log: logging.Logger, label: str):
    t0 = time.perf_counter()
    log.info("START %s", label)
    yield
    log.info("END   %s (%.1fs)", label, time.perf_counter() - t0)
