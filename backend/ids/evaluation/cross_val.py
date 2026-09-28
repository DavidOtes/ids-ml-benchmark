"""Stratified k-fold cross-validation for the model comparison.

The imblearn Pipeline built by ``build_pipeline`` embeds under/over-sampling as
steps, so fitting it on a training fold resamples *only* that fold — the held-out
validation fold is never resampled and never leaks into training. This makes CV a
fair, low-variance estimate of each model's generalization on the combined dataset.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from ids.config import RANDOM_STATE
from ids.models.pipelines import build_pipeline
from ids.models.registry import MODEL_NAMES

log = logging.getLogger(__name__)

# Metric columns produced per (model, fold). Order matters for the output CSV.
METRIC_NAMES = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"]


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  y_proba: np.ndarray | None, classes: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": np.nan,
    }
    if y_proba is not None and y_proba.shape[1] == len(classes):
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr",
                              average="weighted", labels=classes)
            )
        except Exception as exc:  # e.g. a fold missing a class
            log.warning("roc_auc unavailable for fold: %s", exc)
    return metrics


def cross_validate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    selected_features: list[str],
    model_names: Iterable[str] = MODEL_NAMES,
    n_splits: int = 5,
    target_per_class: int | None = None,
) -> tuple[pd.DataFrame, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """Run stratified k-fold CV for each model.

    Returns:
        scores: long-form DataFrame with columns
                [model, fold, accuracy, precision_macro, recall_macro,
                 f1_macro, roc_auc, fit_seconds].
        fold_preds: {model -> [(y_true_fold, y_pred_fold), ...]} for downstream
                    significance testing if desired.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X, y))

    rows: list[dict] = []
    fold_preds: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for name in model_names:
        fold_preds[name] = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y[train_idx]
            y_val = y[val_idx]

            kwargs = {}
            if target_per_class is not None:
                kwargs["target_per_class"] = target_per_class
            pipeline = build_pipeline(name, selected_features=selected_features, **kwargs)

            t0 = time.perf_counter()
            try:
                pipeline.fit(X_tr, y_tr)
            except Exception as exc:
                # e.g. the calibrated SVM needs >=3 samples/class per calibration fold;
                # on a subsample the rarest classes can fall below that. Record NaNs
                # for this fold instead of aborting the whole CV run.
                log.warning("fit failed for %s fold %d (recording NaN): %s", name, fold_idx, exc)
                rows.append({"model": name, "fold": fold_idx,
                             **{m: np.nan for m in METRIC_NAMES}, "fit_seconds": np.nan})
                continue
            fit_seconds = time.perf_counter() - t0

            y_pred = pipeline.predict(X_val)
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_val)
                except Exception as exc:
                    log.warning("predict_proba failed for %s fold %d: %s", name, fold_idx, exc)

            metrics = _fold_metrics(y_val, y_pred, y_proba, pipeline.classes_)
            rows.append({"model": name, "fold": fold_idx, **metrics,
                         "fit_seconds": round(fit_seconds, 3)})
            fold_preds[name].append((y_val, y_pred))
            log.info("CV %s fold %d/%d: f1_macro=%.4f acc=%.4f (%.1fs)",
                     name, fold_idx + 1, n_splits, metrics["f1_macro"],
                     metrics["accuracy"], fit_seconds)

    return pd.DataFrame(rows), fold_preds


def summarize_cv(scores: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fold scores to per-model mean ± std for each metric."""
    agg = scores.groupby("model")[METRIC_NAMES].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()
