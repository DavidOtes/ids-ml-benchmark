from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def read_full(path: Path) -> pd.DataFrame:
    """Read an entire CSV into memory. For 2017 files (each ≤215 MB)."""
    df = pd.read_csv(path, low_memory=False, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def iter_csv_chunks(path: Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Stream a large CSV in chunks. For 2018 files."""
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, skipinitialspace=True):
        chunk.columns = [c.strip() for c in chunk.columns]
        yield chunk


def read_and_sample(
    path: Path,
    rows_per_file: int,
    label_col: str = "Label",
    random_state: int = 42,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read a large CSV in chunks and stratified-sample rows_per_file rows total.

    The sample is drawn proportionally from each chunk so minority attack classes
    survive. If the file has fewer rows than rows_per_file, all rows are returned.
    """
    rng = np.random.default_rng(random_state)
    collected: list[pd.DataFrame] = []

    chunks: list[pd.DataFrame] = []
    total_rows = 0
    for chunk in iter_csv_chunks(path, chunksize=chunksize):
        chunks.append(chunk)
        total_rows += len(chunk)

    if total_rows <= rows_per_file:
        log.info("%s: %d rows <= target %d, returning all", path.name, total_rows, rows_per_file)
        return pd.concat(chunks, ignore_index=True)

    for chunk in chunks:
        share = int(round(rows_per_file * len(chunk) / total_rows))
        if share <= 0:
            continue

        if label_col in chunk.columns:
            try:
                sampled = (
                    chunk.groupby(label_col, group_keys=False, observed=True)
                    .apply(lambda g: g.sample(
                        n=min(len(g), max(1, int(round(share * len(g) / len(chunk))))),
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    ))
                )
            except Exception as exc:
                log.warning("stratified sample failed (%s); falling back to uniform", exc)
                sampled = chunk.sample(n=min(len(chunk), share),
                                       random_state=int(rng.integers(0, 2**31 - 1)))
        else:
            sampled = chunk.sample(n=min(len(chunk), share),
                                   random_state=int(rng.integers(0, 2**31 - 1)))

        collected.append(sampled)

    out = pd.concat(collected, ignore_index=True)
    log.info("%s: sampled %d / %d rows", path.name, len(out), total_rows)
    return out
