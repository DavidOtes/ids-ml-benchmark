"""Hyperparameter tuning via cross-validated search.

Mirrors the reference dissertation's Grid Search Cross-Validation step: each model
is tuned with stratified k-fold CV over a small, sensible grid, scoring on
``f1_macro`` (the primary comparison metric, robust to the class imbalance). The
search runs over the full imblearn pipeline, so under/over-sampling is applied inside
each CV fold only — the tuning never sees resampled validation data.

Grid keys use the pipeline's ``clf__`` prefix, so ``GridSearchCV.best_params_`` can be
fed straight back into ``build_pipeline(clf_params=...)``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from ids.config import RANDOM_STATE, SAMPLER_TARGET_PER_CLASS
from ids.models.pipelines import build_pipeline
from ids.models.registry import make_svm_base

log = logging.getLogger(__name__)

# Per-model search spaces. Kept deliberately small so a full GridSearch is feasible;
# widen these (or switch to --search random) for a more exhaustive study.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "decision_tree": {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [10, 25, None],
        "clf__min_samples_split": [2, 10],
    },
    "random_forest": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 25],
        "clf__max_features": ["sqrt", "log2"],
    },
    "svm": {
        # SGD hinge is wrapped in SingleFitCalibratedClassifier -> nested path.
        # alpha is the L2 strength (roughly 1/(C*n_samples) vs the old C grid).
        "clf__estimator__alpha": [1e-5, 1e-4, 1e-3],
    },
    "naive_bayes": {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
    },
    "knn": {
        "clf__n_neighbors": [3, 5, 7],
        "clf__weights": ["uniform", "distance"],
    },
}


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    selected_features: list[str],
    *,
    scoring: str = "f1_macro",
    cv: int = 3,
    search: str = "grid",
    n_iter: int = 10,
    n_jobs: int = 1,
    target_per_class: int = SAMPLER_TARGET_PER_CLASS,
) -> tuple[dict, float | None]:
    """Cross-validated search for one model.

    Returns ``(best_params, best_score)``. ``best_params`` keys carry the ``clf__``
    prefix. Returns ``({}, None)`` for models with no grid defined.
    """
    grid = PARAM_GRIDS.get(name)
    if not grid:
        log.info("no param grid for %s — skipping tuning", name)
        return {}, None

    pipeline = build_pipeline(name, selected_features, target_per_class=target_per_class)

    # The production SVM wraps the SGD hinge classifier in single-fit Platt
    # calibration (for predict_proba). Calibration only maps scores ->
    # probabilities; it doesn't move the decision boundary, so the best alpha is
    # identical. Tuning the bare classifier skips fitting calibrators the search
    # would immediately discard. We remap the winning alpha back to the
    # calibrated form below so 04_train applies it verbatim.
    if name == "svm":
        pipeline.set_params(clf=make_svm_base())
        grid = {"clf__alpha": PARAM_GRIDS["svm"]["clf__estimator__alpha"]}

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    # n_jobs=1 by default: RF/KNN already parallelise internally, so parallelising the
    # search too would oversubscribe cores. Bump via the script flag on big machines.
    if search == "random":
        searcher = RandomizedSearchCV(
            pipeline, grid, n_iter=n_iter, scoring=scoring, cv=splitter,
            n_jobs=n_jobs, random_state=RANDOM_STATE, refit=True, verbose=1,
        )
    else:
        searcher = GridSearchCV(
            pipeline, grid, scoring=scoring, cv=splitter,
            n_jobs=n_jobs, refit=True, verbose=1,
        )

    searcher.fit(X, y)
    best_params = dict(searcher.best_params_)
    # Remap the bare-classifier alpha onto the calibrated SVM's nested param path.
    if name == "svm" and "clf__alpha" in best_params:
        best_params = {"clf__estimator__alpha": best_params.pop("clf__alpha")}
    log.info("%s best %s=%.4f with %s", name, scoring, searcher.best_score_, best_params)
    return best_params, float(searcher.best_score_)
