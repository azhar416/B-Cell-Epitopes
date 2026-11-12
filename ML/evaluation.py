"""Metrics and plots shared by the ML training workflows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate overall and per-class binary metrics.

    Per-class accuracy is the fraction of actual members of that class
    classified correctly, so it is equal to that class's recall.
    """
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    class_counts = {
        0: (int(tn), int(fn), int(fp), int(tn + fp)),
        1: (int(tp), int(fp), int(fn), int(tp + fn)),
    }
    metrics: dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }
    for class_label, (class_tp, class_fp, class_fn, support) in class_counts.items():
        class_precision = class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0
        class_recall = class_tp / support if support else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        metrics.update(
            {
                f"class_{class_label}_accuracy": float(class_recall),
                f"class_{class_label}_precision": float(class_precision),
                f"class_{class_label}_recall": float(class_recall),
                f"class_{class_label}_f1": float(class_f1),
            }
        )
    return metrics


def save_evaluation_plots(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
    prefix: str = "",
    title_prefix: str = "Validation",
) -> dict[str, str]:
    """Save validation ROC/precision-recall curves and a confusion-matrix heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)
    axes[0].plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {roc_auc:.3f}",
        color="tab:blue",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axes[0].set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{title_prefix} ROC curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)
    axes[1].plot(recall, precision, label=f"Average precision = {pr_auc:.3f}")
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{title_prefix} precision-recall curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[1].legend(loc="best")
    figure.tight_layout()
    curves_filename = f"{prefix}evaluation_curves.png" if prefix else "evaluation_curves.png"
    curves_path = output_dir / curves_filename
    figure.savefig(curves_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{title_prefix} confusion matrix (threshold={threshold:g})",
    )
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else "black",
            )
    figure.colorbar(image, ax=axis, label="Examples")
    figure.tight_layout()
    matrix_filename = f"{prefix}confusion_matrix.png" if prefix else "confusion_matrix.png"
    matrix_path = output_dir / matrix_filename
    figure.savefig(matrix_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    return {
        "evaluation_curves": str(curves_path),
        "confusion_matrix": str(matrix_path),
    }


def save_train_validation_metrics_plot(
    train_metrics: dict[str, float | int],
    validation_metrics: dict[str, float | int],
    output_dir: Path,
    train_label: str = "Training",
    validation_label: str = "Validation",
    title: str = "Training vs validation metrics",
) -> str:
    """Save a comparison of fitted-model training and validation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = (
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Accuracy", "accuracy"),
        ("Recall", "recall"),
        ("Precision", "precision"),
        ("F1", "f1"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("MCC", "mcc"),
    )
    positions = np.arange(len(metric_names))
    width = 0.36
    train_values = [train_metrics[key] for _, key in metric_names]
    validation_values = [validation_metrics[key] for _, key in metric_names]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width / 2, train_values, width, label=train_label)
    axis.bar(
        positions + width / 2,
        validation_values,
        width,
        label=validation_label,
    )
    axis.set(
        ylabel="Score",
        title=title,
        xticks=positions,
        xticklabels=[name for name, _ in metric_names],
        ylim=(-1.05, 1.05),
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / "training_validation_metrics.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(plot_path)
