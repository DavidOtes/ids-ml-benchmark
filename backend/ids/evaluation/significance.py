"""Statistical significance tests for the model comparison.

Two complementary views:

* ``mcnemar_pairwise`` — compares two classifiers on the *same* held-out test set.
  McNemar's test looks only at the discordant pairs (cases where exactly one model
  is right) and asks whether that split is balanced. It answers "do these two models
  make significantly different errors on this test set?".

* ``cv_paired_tests`` — compares two classifiers across the *same* k-fold splits.
  A paired t-test and the (non-parametric) Wilcoxon signed-rank test on per-fold F1
  answer "is one model consistently better across folds?".

All p-values use scipy only (already a dependency).
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, ttest_rel, wilcoxon

log = logging.getLogger(__name__)

# Below this many discordant pairs the chi-square approximation is unreliable,
# so we use the exact binomial test instead.
_EXACT_THRESHOLD = 25


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test on two models' predictions over the same samples.

    b = A correct & B wrong, c = A wrong & B correct. Uses the exact binomial test
    when b + c is small, otherwise the chi-square statistic with continuity correction.
    """
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c

    if n == 0:
        statistic, p_value = 0.0, 1.0
    elif n < _EXACT_THRESHOLD:
        # Exact: two-sided binomial with p=0.5 over the discordant pairs.
        p_value = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
        statistic = float(min(b, c))
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n)  # continuity-corrected chi-square
        p_value = float(chi2.sf(statistic, df=1))

    # "better" = the model correct more often among the discordant pairs.
    # b = A right/B wrong, c = A wrong/B right, so A wins when b > c.
    better = None
    if b != c:
        better = "a" if b > c else "b"
    return {"b_only_a_correct": b, "c_only_b_correct": c,
            "statistic": statistic, "p_value": p_value, "better": better}


def mcnemar_pairwise(predictions_by_model: dict[str, np.ndarray],
                     y_true: np.ndarray) -> pd.DataFrame:
    """Run McNemar for every model pair. Returns a long-form DataFrame."""
    rows = []
    for model_a, model_b in itertools.combinations(predictions_by_model, 2):
        res = mcnemar_test(y_true, predictions_by_model[model_a],
                           predictions_by_model[model_b])
        better_name = {None: "tie", "a": model_a, "b": model_b}[res["better"]]
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "only_a_correct": res["b_only_a_correct"],
            "only_b_correct": res["c_only_b_correct"],
            "statistic": round(res["statistic"], 4),
            "p_value": res["p_value"],
            "significant_0.05": res["p_value"] < 0.05,
            "better": better_name,
        })
    return pd.DataFrame(rows)


def cv_paired_tests(cv_scores: pd.DataFrame, metric: str = "f1_macro") -> pd.DataFrame:
    """Paired t-test + Wilcoxon signed-rank on per-fold ``metric`` for each model pair.

    ``cv_scores`` is the long-form frame from ``cross_validate_models`` with columns
    [model, fold, <metric>, ...].
    """
    pivot = cv_scores.pivot(index="fold", columns="model", values=metric)
    rows = []
    for model_a, model_b in itertools.combinations(pivot.columns, 2):
        a = pivot[model_a].to_numpy()
        b = pivot[model_b].to_numpy()
        diff = a - b

        t_stat, t_p = ttest_rel(a, b)
        # Wilcoxon errors when all differences are zero; treat that as p=1.
        if np.allclose(diff, 0):
            w_stat, w_p = 0.0, 1.0
        else:
            try:
                w_stat, w_p = wilcoxon(a, b)
            except ValueError as exc:
                log.warning("wilcoxon fallback for %s vs %s: %s", model_a, model_b, exc)
                w_stat, w_p = np.nan, 1.0

        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "mean_a": round(float(a.mean()), 4),
            "mean_b": round(float(b.mean()), 4),
            "mean_diff": round(float(diff.mean()), 4),
            "t_stat": round(float(t_stat), 4),
            "t_p_value": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p_value": float(w_p),
            "significant_0.05": bool(t_p < 0.05),
        })
    return pd.DataFrame(rows)
