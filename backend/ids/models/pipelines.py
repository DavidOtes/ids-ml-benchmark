from __future__ import annotations

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from ids.config import SAMPLER_TARGET_PER_CLASS, SMOTE_K_NEIGHBORS
from ids.models.registry import MODELS
from ids.preprocess.balance import make_over_sampler, make_under_sampler


def _make_column_selector(selected_features: list[str]) -> ColumnTransformer:
    """Select + order the chosen features and pass them through unchanged.

    Wrapping in a ColumnTransformer means the fitted Pipeline reads input
    DataFrames by column name, so callers may pass any superset of columns
    (e.g. the full canonical frame in the cross-dataset study).
    """
    return ColumnTransformer(
        transformers=[("selected", "passthrough", selected_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    model_name: str,
    selected_features: list[str],
    target_per_class: int = SAMPLER_TARGET_PER_CLASS,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    clf_params: dict | None = None,
) -> ImbPipeline:
    """Build the imblearn pipeline for ``model_name``.

    ``clf_params`` overrides classifier hyperparameters. Keys use the pipeline's
    ``clf__`` prefix exactly as produced by GridSearchCV.best_params_ (e.g.
    ``{"clf__max_depth": 25}``), so tuned params from scripts/03b_tune.py can be
    applied verbatim.
    """
    if model_name not in MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")
    steps = [
        ("select", _make_column_selector(selected_features)),
        ("scale", StandardScaler()),
        ("under", make_under_sampler(target_per_class)),
        ("over", make_over_sampler(target_per_class, k_neighbors)),
        ("clf", MODELS[model_name]()),
    ]
    pipeline = ImbPipeline(steps=steps)
    if clf_params:
        pipeline.set_params(**clf_params)
    return pipeline
