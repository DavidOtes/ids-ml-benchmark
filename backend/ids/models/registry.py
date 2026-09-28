from __future__ import annotations

from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ids.config import RANDOM_STATE
from ids.models.calibration import SingleFitCalibratedClassifier

MODEL_NAMES: list[str] = [
    "decision_tree",
    "random_forest",
    "svm",
    "naive_bayes",
    "knn",
]


def make_svm_base() -> SGDClassifier:
    """Linear SVM (hinge loss + L2) trained by SGD.

    Replaces LinearSVC: liblinear never converged within max_iter=2000 on the
    SMOTE-balanced data, so every one-vs-rest subproblem burned the full
    iteration budget. SGD with early stopping reaches the boundary in a few
    passes and scales linearly with rows. Shared with scripts/03b_tune so
    tuning and training use identical solver settings.
    """
    return SGDClassifier(
        loss="hinge",
        random_state=RANDOM_STATE,
        tol=1e-3,
        early_stopping=True,
        n_iter_no_change=3,
        average=True,
        n_jobs=-1,
    )


MODELS: dict[str, Callable[[], BaseEstimator]] = {
    "decision_tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=25),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    ),
    "svm": lambda: SingleFitCalibratedClassifier(
        estimator=make_svm_base(),
        calibration_size=0.15,
        random_state=RANDOM_STATE,
    ),
    "naive_bayes": lambda: GaussianNB(),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}
