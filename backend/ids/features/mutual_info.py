"""Mutual-information (information gain) feature ranking — filter method.

Ranks each feature by its estimated mutual information with the class label
(Kraskov k-NN estimator via sklearn). Model-free and sensitive to non-linear
feature-class relationships, unlike variance- or correlation-based filters.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from ids.config import RANDOM_STATE
from ids.features.subsample import stratified_subsample

log = logging.getLogger(__name__)


def rank_by_mutual_info(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_rows: int = 300_000,
    random_state: int = RANDOM_STATE,
) -> list[tuple[str, float]]:
    """Return [(feature, mi_score), ...] sorted best-first."""
    Xs, ys = stratified_subsample(X, y, sample_rows, random_state=random_state)
    log.info("mutual information on %d rows x %d features", len(Xs), Xs.shape[1])
    scores = mutual_info_classif(Xs, ys, random_state=random_state, n_jobs=-1)
    pairs = list(zip(X.columns.tolist(), [float(s) for s in scores]))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs
