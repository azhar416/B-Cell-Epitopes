"""Metrics and plots for binary epitope classification."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional.")
    if len(labels) != len(probabilities) or len(labels) == 0:
        raise ValueError("labels and probabilities must be non-empty and aligned.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if np.unique(labels).size != 2:
        raise ValueError("Evaluation requires examples of both label classes.")
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_prevalence": float(labels.mean()),
    }


def save_evaluation_plots(
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_dir: str | Path,
    threshold: float,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    axes[0].plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC={roc_auc_score(labels, probabilities):.3f}",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[0].legend()
    axes[1].plot(
        recall,
        precision,
        label=f"PR-AUC={average_precision_score(labels, probabilities):.3f}",
    )
    axes[1].axhline(
        float(np.mean(labels)),
        linestyle="--",
        color="gray",
        label="Prevalence",
    )
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[1].legend()
    figure.tight_layout()
    curves_path = output_dir / "evaluation_curves.png"
    figure.savefig(curves_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5, 5))
    matrix = np.asarray([[tn, fp], [fn, tp]])
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=(0, 1),
        yticks=(0, 1),
        xticklabels=("Negative", "Positive"),
        yticklabels=("Negative", "Positive"),
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion matrix (threshold={threshold:g})",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    confusion_path = output_dir / "confusion_matrix.png"
    figure.savefig(confusion_path, dpi=160)
    plt.close(figure)
    return {
        "evaluation_curves": str(curves_path),
        "confusion_matrix": str(confusion_path),
    }
