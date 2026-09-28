from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc: Optional[float]
    detection_time_ms: float
    n_test: int

    def to_dict(self) -> dict:
        return asdict(self)


def _roc_auc_ovr(pipeline, X, y_true) -> Optional[float]:
    try:
        if not hasattr(pipeline, "predict_proba"):
            return None
        proba = pipeline.predict_proba(X)
        classes = pipeline.classes_
        # Re-index y to the class list to avoid label-mismatch errors
        if proba.shape[1] != len(classes):
            return None
        return float(roc_auc_score(y_true, proba, multi_class="ovr",
                                   average="weighted", labels=classes))
    except Exception as exc:
        log.warning("roc_auc fallback (returned None): %s", exc)
        return None


def evaluate_model(
    name: str,
    pipeline,
    X_test,
    y_test: np.ndarray,
    timing_sample_size: int = 10_000,
) -> tuple[ModelMetrics, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (metrics, confusion_matrix, y_pred, y_proba).

    y_proba is None when the underlying estimator does not expose predict_proba.
    """
    y_pred = pipeline.predict(X_test)

    y_proba: np.ndarray | None = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)
        except Exception as exc:
            log.warning("predict_proba failed on %s: %s", name, exc)
            y_proba = None

    n = min(timing_sample_size, len(X_test))
    X_small = X_test.iloc[:n] if hasattr(X_test, "iloc") else X_test[:n]
    t0 = time.perf_counter()
    pipeline.predict(X_small)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    per_sample_ms = elapsed_ms / max(n, 1)

    metrics = ModelMetrics(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        roc_auc=_roc_auc_ovr(pipeline, X_test, y_test),
        detection_time_ms=float(per_sample_ms),
        n_test=int(len(X_test)),
    )
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_pred, y_proba
