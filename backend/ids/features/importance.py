from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ids.config import RANDOM_STATE


def rank_by_rf_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int = 100,
) -> list[tuple[str, float]]:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y)
    pairs = list(zip(X.columns.tolist(), rf.feature_importances_.tolist()))
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs


def select_top_k(ranking: list[tuple[str, float]], k: int) -> list[str]:
    return [name for name, _ in ranking[:k]]
