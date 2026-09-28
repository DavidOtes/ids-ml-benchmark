from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATASETS_DIR = Path(os.environ.get("DATASETS_DIR", PROJECT_ROOT / "datasets")).resolve()
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", BACKEND_DIR / "artifacts")).resolve()

CIC2017_DIR = DATASETS_DIR / "CIC-IDS- 2017"
CIC2018_DIR = DATASETS_DIR / "CSE-CIC-IDS2018"

INTERIM_DIR = ARTIFACTS_DIR / "interim"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
LOGS_DIR = ARTIFACTS_DIR / "logs"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_K_FEATURES = 30
CORRELATION_THRESHOLD = 0.95
DETECTION_TIME_SAMPLE_SIZE = 10_000
# Target rows per class after sampling (bounds memory during training).
# Classes with more rows are under-sampled; classes with fewer rows are SMOTE'd up.
SAMPLER_TARGET_PER_CLASS = 50_000
SMOTE_K_NEIGHBORS = 5

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, METRICS_DIR, LOGS_DIR,
              METRICS_DIR / "plots", METRICS_DIR / "confusion_matrices",
              METRICS_DIR / "cv", METRICS_DIR / "significance",
              METRICS_DIR / "cross_dataset"):
        d.mkdir(parents=True, exist_ok=True)


# --- Per-dataset artifact layout ---------------------------------------------
# When training each dataset separately (no combining, no feature selection),
# every stage writes under artifacts/<dataset>/ so the two studies never share
# models, label encoders, or metrics. The interim samples remain shared because
# 01_build_sample.py already isolates them by year (cic2017_sample.parquet etc.).

VALID_DATASETS = ("cic2017", "cic2018")


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved artifact subtree for one dataset (or the combined run)."""

    name: str
    interim: Path
    processed: Path
    models: Path
    metrics: Path
    logs: Path


def dataset_paths(name: str | None) -> DatasetPaths:
    """Return the artifact paths for ``name``.

    ``name=None`` yields the original combined-run layout (backward compatible),
    so scripts default to the legacy dirs unless ``--dataset`` is passed.
    """
    if name is None:
        return DatasetPaths("combined", INTERIM_DIR, PROCESSED_DIR, MODELS_DIR,
                            METRICS_DIR, LOGS_DIR)
    if name not in VALID_DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {VALID_DATASETS}")
    root = ARTIFACTS_DIR / name
    return DatasetPaths(name, INTERIM_DIR, root / "processed", root / "models",
                        root / "metrics", root / "logs")


def ensure_dataset_dirs(name: str | None) -> None:
    """Create the per-dataset subtree (or the combined dirs when ``name=None``)."""
    if name is None:
        ensure_dirs()
        return
    p = dataset_paths(name)
    for d in (p.processed, p.models, p.metrics, p.logs,
              p.metrics / "plots", p.metrics / "confusion_matrices",
              p.metrics / "per_class", p.metrics / "predictions", p.metrics / "probas",
              p.metrics / "cv", p.metrics / "significance"):
        d.mkdir(parents=True, exist_ok=True)
