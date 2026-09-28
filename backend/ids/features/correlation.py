from __future__ import annotations

import numpy as np
import pandas as pd


def drop_correlated(X: pd.DataFrame, threshold: float = 0.95) -> tuple[list[str], list[str]]:
    """Return (kept_columns, dropped_columns) using a |corr| > threshold filter.

    For any pair of features with absolute correlation above the threshold, the
    one appearing later in column order is dropped.
    """
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    dropped = [col for col in upper.columns if any(upper[col] > threshold)]
    kept = [c for c in X.columns if c not in dropped]
    return kept, dropped
