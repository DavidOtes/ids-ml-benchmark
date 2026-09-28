"""SVM-RFE feature ranking — wrapper method (Guyon et al., 2002).

Recursive Feature Elimination around the study's linear SVM (SGD hinge):
repeatedly fit, drop the ``step`` features with the smallest coefficient
magnitudes, and refit, until ``n_select`` remain. Features are scaled first
(the SVM is margin-based), fitted on a stratified subsample of the training
data only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

from ids.config import RANDOM_STATE
from ids.features.subsample import stratified_subsample
from ids.models.registry import make_svm_base

log = logging.getLogger(__name__)


def rank_by_svm_rfe(
    X: pd.DataFrame,
    y: np.ndarray,
    n_select: int = 40,
    step: int = 5,
    sample_rows: int = 300_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a DataFrame [feature, selected, rfe_rank, coef_mag] sorted best-first.

    ``rfe_rank`` is 1 for every selected feature (RFE does not order within the
    survivors), so survivors are ordered by mean |coefficient| in the final
    fit; eliminated features follow, in reverse order of elimination.
    """
    Xs, ys = stratified_subsample(X, y, sample_rows, random_state=random_state)
    Xs_sc = StandardScaler().fit_transform(Xs)
    log.info("SVM-RFE on %d rows x %d features (step=%d, target=%d)",
             len(Xs), Xs.shape[1], step, n_select)
    rfe = RFE(make_svm_base(), n_features_to_select=n_select, step=step)
    rfe.fit(Xs_sc, ys)

    coef_mag = np.abs(rfe.estimator_.coef_).mean(axis=0)  # final fit, selected only
    mags = dict(zip(np.array(X.columns)[rfe.support_], coef_mag))
    out = pd.DataFrame({
        "feature": X.columns,
        "selected": rfe.support_,
        "rfe_rank": rfe.ranking_,
        "coef_mag": [round(float(mags.get(c, 0.0)), 6) for c in X.columns],
    })
    out = out.sort_values(["rfe_rank", "coef_mag"], ascending=[True, False],
                          ignore_index=True)
    return out
