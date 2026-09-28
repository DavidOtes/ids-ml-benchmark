from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ids.schema import CANONICAL_COLUMNS, LABEL_COLUMN

log = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace inf with NaN, drop NaN rows, drop duplicates.

    Returns the cleaned DataFrame and a stats dict useful for logging.
    """
    stats: dict[str, int] = {"input_rows": len(df)}

    feature_cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    stats["replaced_inf"] = int(np.isinf(df[feature_cols].to_numpy()).sum())
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=feature_cols + [LABEL_COLUMN])
    stats["dropped_nan"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    stats["dropped_duplicates"] = before - len(df)

    stats["output_rows"] = len(df)
    log.info("cleaning stats: %s", stats)
    return df.reset_index(drop=True), stats
