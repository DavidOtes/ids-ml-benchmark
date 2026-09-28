"""Single-fit probability calibration for margin classifiers.

``CalibratedClassifierCV(cv=k)`` refits the base estimator k times and keeps
the whole ensemble — for the SVM that tripled training cost. This wrapper
fits the base estimator ONCE on a stratified share of the training data, then
Platt-scales the frozen model on the held-out remainder (``cv="prefit"``).
The result still exposes ``predict_proba``, so ROC-AUC and the probability
artifacts are unaffected.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


class SingleFitCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Fit ``estimator`` once, then sigmoid-calibrate it on a held-out slice.

    ``estimator`` is exposed as a nested sklearn param, so tuned
    hyperparameters keep the ``clf__estimator__<param>`` path shape written by
    scripts/03b_tune.py to best_params.json.

    The stratified calibration split requires every class to have at least 2
    training samples — the SMOTE-balanced pipeline guarantees far more.
    """

    def __init__(self, estimator, calibration_size: float = 0.15,
                 random_state: int | None = None):
        self.estimator = estimator
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X, y):
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y,
            test_size=self.calibration_size,
            stratify=y,
            random_state=self.random_state,
        )
        self.estimator_ = clone(self.estimator).fit(X_fit, y_fit)
        self.calibrated_ = CalibratedClassifierCV(self.estimator_, cv="prefit")
        self.calibrated_.fit(X_cal, y_cal)
        self.classes_ = self.calibrated_.classes_
        return self

    def predict(self, X):
        return self.calibrated_.predict(X)

    def predict_proba(self, X):
        return self.calibrated_.predict_proba(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)
