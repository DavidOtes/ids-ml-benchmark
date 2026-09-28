"""Class-proportional subsampling for feature-selection fitting.

Selection methods (mutual information, RFE) don't need millions of rows to
rank 77 features; a stratified subsample keeps the ranking stable while making
the fit fast. Rare classes are never dropped below ``min_per_class`` rows (or
their full count), so downstream stratified splits keep working.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ids.config import RANDOM_STATE

log = logging.getLogger(__name__)


def stratified_subsample(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int,
    min_per_class: int = 8,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(random_state)
    frac = max_rows / len(X)
    positions = []
    for cls in np.unique(y):
        pos = np.flatnonzero(y == cls)
        n = max(min(len(pos), min_per_class), int(round(len(pos) * frac)))
        positions.append(rng.choice(pos, size=min(n, len(pos)), replace=False))
    idx = np.concatenate(positions)
    rng.shuffle(idx)
    log.info("stratified subsample: %d -> %d rows (%d classes preserved)",
             len(X), len(idx), len(np.unique(y)))
    return X.iloc[idx], y[idx]
