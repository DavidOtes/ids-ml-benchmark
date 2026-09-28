from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def fit_label_encoder(labels: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(labels.astype(str))
    return encoder


def save_label_encoder(encoder: LabelEncoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, path)


def load_label_encoder(path: Path) -> LabelEncoder:
    return joblib.load(path)


def transform_labels(encoder: LabelEncoder, labels: pd.Series) -> np.ndarray:
    return encoder.transform(labels.astype(str))
