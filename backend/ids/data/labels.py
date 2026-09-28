from __future__ import annotations

import pandas as pd

BENIGN_LABEL = "Benign"


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Strip whitespace and unify benign casing across the two datasets.

    2017 uses "BENIGN"; 2018 uses "Benign". Attack-type labels are kept as-is
    (after trimming whitespace) so multi-class classification sees every subtype.
    """
    cleaned = labels.astype(str).str.strip()
    cleaned = cleaned.replace({"BENIGN": BENIGN_LABEL, "Benign": BENIGN_LABEL})
    return cleaned


def is_attack_series(labels: pd.Series) -> pd.Series:
    return normalize_labels(labels) != BENIGN_LABEL
