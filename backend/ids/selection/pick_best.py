from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BestPick:
    name: str
    score: float
    components: dict[str, float]


def composite_score(f1: float, roc_auc: float | None, detection_time_ms: float,
                    max_detection_time_ms: float) -> float:
    """0.5·F1 + 0.3·ROC-AUC + 0.2·(1 − normalized_time).

    ROC-AUC falls back to F1 when a model couldn't produce probabilities.
    Detection time is min-max normalized against the slowest model in the cohort.
    """
    auc = roc_auc if roc_auc is not None else f1
    speed = 1.0 - (detection_time_ms / max_detection_time_ms) if max_detection_time_ms > 0 else 1.0
    return 0.5 * f1 + 0.3 * auc + 0.2 * speed


def pick_best(metrics: dict[str, dict]) -> BestPick:
    max_time = max(m["detection_time_ms"] for m in metrics.values()) or 1.0
    ranked: list[BestPick] = []
    for name, m in metrics.items():
        score = composite_score(
            f1=m["f1_macro"],
            roc_auc=m.get("roc_auc"),
            detection_time_ms=m["detection_time_ms"],
            max_detection_time_ms=max_time,
        )
        ranked.append(BestPick(
            name=name,
            score=score,
            components={
                "f1_macro": m["f1_macro"],
                "roc_auc": m.get("roc_auc"),
                "detection_time_ms": m["detection_time_ms"],
            },
        ))
    ranked.sort(key=lambda b: b.score, reverse=True)
    return ranked[0]
