from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from ids.evaluation.metrics import ModelMetrics


def write_metrics_json(
    metrics_by_model: dict[str, ModelMetrics],
    dataset_summary: dict,
    out_path: Path,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_summary,
        "models": {name: m.to_dict() for name, m in metrics_by_model.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def save_confusion_matrix(name: str, cm: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", cm)


def write_summary_csv(metrics_by_model: dict[str, ModelMetrics], out_path: Path) -> None:
    """One-row-per-model table of scalar metrics — the canonical CSV for plotting."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([m.to_dict() for m in metrics_by_model.values()]).to_csv(out_path, index=False)


def write_per_class_csv(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    """Per-class precision / recall / F1 / support from sklearn's classification_report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls_name in class_names:
        row = report.get(cls_name)
        if row is None:
            continue
        rows.append({"class": cls_name, **row})
    for summary_key in ("macro avg", "weighted avg"):
        if summary_key in report:
            rows.append({"class": summary_key, **report[summary_key]})
    pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)


def write_predictions_parquet(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "y_true": y_true.astype(np.int32),
        "y_pred": y_pred.astype(np.int32),
    }).to_parquet(out_dir / f"{name}.parquet", index=False)


def write_probas_parquet(
    name: str,
    proba: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    """Full class-probability matrix. Enables ROC / PR / calibration plots later."""
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"p_{c}" for c in class_names[: proba.shape[1]]]
    pd.DataFrame(proba, columns=columns).to_parquet(out_dir / f"{name}.parquet", index=False)
