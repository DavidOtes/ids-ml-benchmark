from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from ids.schema import (
    CANONICAL_COLUMNS,
    DROP_2017_DUPLICATES,
    DROP_2018_ONLY,
    LABEL_COLUMN,
    RENAME_2017,
    RENAME_2018,
)

log = logging.getLogger(__name__)


def _apply_rename(df: pd.DataFrame, rename_map: dict[str, str], drop_cols: Iterable[str]) -> pd.DataFrame:
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.rename(columns=rename_map)
    keep = [c for c in CANONICAL_COLUMNS + [LABEL_COLUMN] if c in df.columns]
    missing = set(CANONICAL_COLUMNS + [LABEL_COLUMN]) - set(keep)
    if missing:
        raise ValueError(f"Unified DataFrame missing expected columns: {sorted(missing)}")
    return df[keep].copy()


def unify_2017(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_rename(df, RENAME_2017, DROP_2017_DUPLICATES)


def unify_2018(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_rename(df, RENAME_2018, DROP_2018_ONLY)


def concat_unified(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("concat_unified: no frames provided")
    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    for col in CANONICAL_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # Rows where every canonical feature is NaN are leaked header rows in the 2018 CSVs.
    mask = out[CANONICAL_COLUMNS].notna().any(axis=1)
    out = out.loc[mask].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        log.info("concat_unified: dropped %d non-numeric rows (likely duplicated headers)", dropped)
    log.info("concat_unified: %d rows, %d columns", len(out), out.shape[1])
    return out
