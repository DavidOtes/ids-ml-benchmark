from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ids.evaluation.metrics import ModelMetrics


_METRIC_FIELDS = [
    ("accuracy", "Accuracy"),
    ("precision_macro", "Precision (macro)"),
    ("recall_macro", "Recall (macro)"),
    ("f1_macro", "F1 (macro)"),
]

# Black-and-white-friendly encodings (the report is printed in greyscale).
_GREYS = ["0.20", "0.45", "0.62", "0.78", "0.90"]
_HATCHES = ["", "//", "..", "xx", "\\\\"]
_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
_MARKERS = ["o", "s", "^", "D", "v"]


def _grey(i: int) -> str:
    return _GREYS[i % len(_GREYS)]


def _hatch(i: int) -> str:
    return _HATCHES[i % len(_HATCHES)]


def title_to_filename(title: str) -> str:
    """Use a plot's (former) title as its image file name.

    Titles are no longer drawn on the figures — the descriptive text lives in the
    file name instead. Only path-hostile characters are replaced.
    """
    name = title.replace("/", "-").replace("\\", "-").strip()
    return f"{name}.png"


def plot_metric_bars(metrics_by_model: dict[str, ModelMetrics], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(metrics_by_model.keys())

    for field, title in _METRIC_FIELDS:
        values = [getattr(metrics_by_model[n], field) for n in names]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(names, values, color="0.6", edgecolor="black", linewidth=0.6)
        ax.set_ylim(0, 1)
        ax.set_ylabel(title)
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / title_to_filename(title), dpi=120)
        plt.close(fig)

    # Inference time on log scale
    title = "Inference time per sample (ms, log scale)"
    times = [metrics_by_model[n].detection_time_ms for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, times, color="0.6", edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("ms/sample")
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)


def plot_confusion_matrix(name: str, cm: np.ndarray, class_names: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Confusion matrix — {name}"
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.4),
                                    max(5, len(class_names) * 0.4)))
    im = ax.imshow(cm, cmap="Greys")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)


def plot_metric_comparison_grouped(metrics_by_model: dict[str, ModelMetrics],
                                   out_dir: Path) -> None:
    """All models × all quality metrics on a single grouped bar chart."""
    title = "Model comparison across metrics"
    names = list(metrics_by_model.keys())
    fields = [f for f, _ in _METRIC_FIELDS]
    labels = [t for _, t in _METRIC_FIELDS]

    x = np.arange(len(fields))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        values = [getattr(metrics_by_model[name], f) for f in fields]
        ax.bar(x + i * width, values, width, label=name, color=_grey(i),
               hatch=_hatch(i), edgecolor="black", linewidth=0.6)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)


def plot_cv_metric_bars(cv_summary: pd.DataFrame, out_dir: Path,
                        metrics: list[str] | None = None) -> None:
    """Grouped bars of CV mean ± std per metric across models.

    ``cv_summary`` is the frame from ``summarize_cv`` (columns like
    ``f1_macro_mean`` / ``f1_macro_std`` plus a ``model`` column).
    """
    title = "Cross-validation metrics by model"
    metrics = metrics or ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metrics = [m for m in metrics if f"{m}_mean" in cv_summary.columns]
    names = cv_summary["model"].tolist()

    x = np.arange(len(metrics))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(cv_summary.iterrows()):
        means = [row[f"{m}_mean"] for m in metrics]
        stds = [row[f"{m}_std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=row["model"],
               color=_grey(i), hatch=_hatch(i), edgecolor="black", linewidth=0.6)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (mean ± std over folds)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)


def plot_roc_overlay(scores_by_model: dict[str, tuple[np.ndarray, np.ndarray]],
                     out_dir: Path) -> None:
    """Overlay every model's attack-vs-benign ROC curve on one axes.

    ``scores_by_model`` maps model -> (y_true_binary, attack_score).
    """
    from sklearn.metrics import auc, roc_curve

    title = "ROC overlay (attack vs benign)"
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, (y_bin, score)) in enumerate(scores_by_model.items()):
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", color="black",
                linestyle=_LINESTYLES[i % len(_LINESTYLES)], linewidth=1.3,
                marker=_MARKERS[i % len(_MARKERS)], markevery=max(1, len(fpr) // 12), markersize=4)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)


def plot_pr_overlay(scores_by_model: dict[str, tuple[np.ndarray, np.ndarray]],
                    out_dir: Path) -> None:
    """Overlay every model's attack-vs-benign precision-recall curve."""
    from sklearn.metrics import precision_recall_curve

    title = "Precision-Recall overlay (attack vs benign)"
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, (y_bin, score)) in enumerate(scores_by_model.items()):
        if y_bin.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin, score)
        ax.plot(recall, precision, label=name, color="black",
                linestyle=_LINESTYLES[i % len(_LINESTYLES)], linewidth=1.3,
                marker=_MARKERS[i % len(_MARKERS)], markevery=max(1, len(recall) // 12), markersize=4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)


def plot_label_distribution(class_distribution: dict[str, int], out_dir: Path) -> None:
    title = "Class distribution (cleaned dataset)"
    items = sorted(class_distribution.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    counts = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.5), 4.5))
    ax.bar(labels, counts, color="0.6", edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("rows (log scale)")
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)


def plot_preprocessing_stats(stats: dict, out_dir: Path) -> None:
    """Bar chart of rows kept vs dropped by reason, from preprocess_stats.json."""
    title = "Preprocessing rows in, removed, out"
    keys = [("input_rows", "input"), ("replaced_inf", "inf replaced"),
            ("dropped_nan", "dropped NaN"), ("dropped_duplicates", "dropped dup"),
            ("output_rows", "output")]
    labels = [lbl for k, lbl in keys if k in stats]
    values = [stats[k] for k, _ in keys if k in stats]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color="0.6", edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("rows (log scale)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)


def plot_significance_heatmap(mcnemar_df: pd.DataFrame, out_dir: Path) -> None:
    """Symmetric model × model heatmap of McNemar p-values."""
    title = "McNemar p-values (below 0.05 = significant difference)"
    models = sorted(set(mcnemar_df["model_a"]) | set(mcnemar_df["model_b"]))
    idx = {m: i for i, m in enumerate(models)}
    mat = np.full((len(models), len(models)), np.nan)
    for _, r in mcnemar_df.iterrows():
        i, j = idx[r["model_a"]], idx[r["model_b"]]
        mat[i, j] = mat[j, i] = r["p_value"]

    fig, ax = plt.subplots(figsize=(1.2 * len(models) + 2, 1.0 * len(models) + 1.5))
    im = ax.imshow(mat, cmap="Greys", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(models)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1e}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="p-value")
    fig.tight_layout()
    fig.savefig(out_dir / title_to_filename(title), dpi=120)
    plt.close(fig)
