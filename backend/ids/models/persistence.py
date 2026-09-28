from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import sklearn
from imblearn.pipeline import Pipeline as ImbPipeline

from ids.models.registry import MODEL_NAMES

log = logging.getLogger(__name__)


def save_pipeline(pipeline: ImbPipeline, name: str, models_dir: Path, meta: dict | None = None) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, path, compress=3)
    meta_payload = {
        "name": name,
        "sklearn_version": sklearn.__version__,
        **(meta or {}),
    }
    (models_dir / f"{name}.meta.json").write_text(json.dumps(meta_payload, indent=2))
    log.info("saved %s", path)
    return path


def load_pipeline(name: str, models_dir: Path) -> ImbPipeline:
    return joblib.load(models_dir / f"{name}.joblib")


def load_all_pipelines(models_dir: Path) -> dict[str, ImbPipeline]:
    out: dict[str, ImbPipeline] = {}
    for name in MODEL_NAMES:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            out[name] = joblib.load(path)
    return out
