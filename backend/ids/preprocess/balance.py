from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from ids.config import RANDOM_STATE, SAMPLER_TARGET_PER_CLASS, SMOTE_K_NEIGHBORS


@dataclass
class UnderTargetStrategy:
    """Cap any class with more than `target_per_class` rows down to the target.

    Implemented as a picklable dataclass so the fitted pipeline can be saved to
    joblib (closures cannot be pickled by stdlib pickle).
    """

    target_per_class: int

    def __call__(self, y):
        counts = Counter(y.tolist() if hasattr(y, "tolist") else list(y))
        return {cls: min(self.target_per_class, cnt)
                for cls, cnt in counts.items() if cnt > self.target_per_class}


@dataclass
class OverTargetStrategy:
    """SMOTE classes below target up to target — but skip classes too rare for SMOTE.

    SMOTE needs at least k_neighbors+1 minority samples. Classes smaller than that
    are kept at their original count (SMOTE can't interpolate from nothing).
    """

    target_per_class: int
    k_neighbors: int

    def __call__(self, y):
        counts = Counter(y.tolist() if hasattr(y, "tolist") else list(y))
        return {
            cls: self.target_per_class
            for cls, cnt in counts.items()
            if cnt < self.target_per_class and cnt > self.k_neighbors
        }


def make_under_sampler(target_per_class: int = SAMPLER_TARGET_PER_CLASS) -> RandomUnderSampler:
    return RandomUnderSampler(
        sampling_strategy=UnderTargetStrategy(target_per_class),
        random_state=RANDOM_STATE,
    )


def make_over_sampler(
    target_per_class: int = SAMPLER_TARGET_PER_CLASS,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
) -> SMOTE:
    return SMOTE(
        sampling_strategy=OverTargetStrategy(target_per_class, k_neighbors),
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors,
    )
