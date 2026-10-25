"""Train and evaluate a classical peptide classifier with a stratified holdout."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
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
    roc_curve,
    roc_auc_score,
)

from ML.CFG import CFG
from ML.dataset import load_labeled_dataset, make_stratified_split
from ML.model import build_model
from ML.pre_dataset import build_feature_frame


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate threshold-dependent and threshold-independent binary metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_evaluation_plots(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_dir: Path,
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
        title="Validation ROC curve",
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
        title="Validation precision-recall curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[1].legend(loc="best")
    figure.tight_layout()
    curves_path = output_dir / "evaluation_curves.png"
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
        title=f"Validation confusion matrix (threshold={threshold:g})",
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
    matrix_path = output_dir / "confusion_matrix.png"
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
) -> str:
    """Save a comparison of fitted-model training and validation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = (
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("F1", "f1"),
        ("MCC", "mcc"),
    )
    positions = np.arange(len(metric_names))
    width = 0.36
    train_values = [train_metrics[key] for _, key in metric_names]
    validation_values = [validation_metrics[key] for _, key in metric_names]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width / 2, train_values, width, label="Training")
    axis.bar(
        positions + width / 2,
        validation_values,
        width,
        label="Validation",
    )
    axis.set(
        ylabel="Score",
        title="Training vs validation metrics",
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


def run_training(config: CFG) -> dict[str, object]:
    """Train on a stratified 80% subset and evaluate on held-out validation data."""
    config.validate()
    dataframe = load_labeled_dataset(
        dataset_name=config.DATASET_NAME,
        data_dir=config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
    )
    feature_frame = build_feature_frame(dataframe, feature_set=config.FEATURE_SET)
    labels = dataframe["target"].to_numpy(dtype=int)
    train_indices, validation_indices = make_stratified_split(
        labels=labels,
        validation_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
    )
    model = build_model(
        model_type=config.MODEL_TYPE,
        random_state=config.RANDOM_STATE,
        random_forest_trees=config.RANDOM_FOREST_TREES,
    )
    model.fit(feature_frame.iloc[train_indices], labels[train_indices])
    train_probabilities = model.predict_proba(
        feature_frame.iloc[train_indices]
    )[:, 1]
    train_metrics = calculate_metrics(
        labels[train_indices],
        train_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
    )
    validation_probabilities = model.predict_proba(
        feature_frame.iloc[validation_indices]
    )[:, 1]
    validation_metrics = calculate_metrics(
        labels[validation_indices],
        validation_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
    )
    comparison_plot = save_train_validation_metrics_plot(
        train_metrics,
        validation_metrics,
        config.OUTPUT_DIR,
    )
    evaluation_plots = save_evaluation_plots(
        labels[validation_indices],
        validation_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
        output_dir=config.OUTPUT_DIR,
    )
    print(
        f"Train: {len(train_indices):,} ({len(train_indices) / len(dataframe):.0%}); "
        f"validation: {len(validation_indices):,} "
        f"({len(validation_indices) / len(dataframe):.0%})"
    )
    print(
        f"Training ROC-AUC={train_metrics['roc_auc']:.4f}, "
        f"PR-AUC={train_metrics['pr_auc']:.4f}, "
        f"F1={train_metrics['f1']:.4f}"
    )
    print(
        f"Validation ROC-AUC={validation_metrics['roc_auc']:.4f}, "
        f"PR-AUC={validation_metrics['pr_auc']:.4f}, "
        f"F1={validation_metrics['f1']:.4f}"
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "peptide_seq": dataframe.iloc[validation_indices]["peptide_seq"].to_numpy(),
            "target": labels[validation_indices],
            "parent_protein_id": dataframe.iloc[validation_indices][
                "parent_protein_id"
            ].to_numpy(),
            "validation_probability": validation_probabilities,
            "validation_prediction": (
                validation_probabilities >= config.PREDICTION_THRESHOLD
            ).astype(int),
        }
    ).to_csv(config.OUTPUT_DIR / "validation_predictions.csv", index=False)
    bundle = {
        "model": model,
        "feature_names": feature_frame.columns.tolist(),
        "feature_set": config.FEATURE_SET,
        "prediction_threshold": config.PREDICTION_THRESHOLD,
        "dataset_name": config.DATASET_NAME,
        "validation_size": config.VALIDATION_SIZE,
        "random_state": config.RANDOM_STATE,
    }
    dump(bundle, config.OUTPUT_DIR / "peptide_classifier.joblib")

    report: dict[str, object] = {
        "dataset": config.DATASET_NAME,
        "model": config.MODEL_TYPE,
        "feature_set": config.FEATURE_SET,
        "split_strategy": "stratified_by_label",
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "train_fraction": len(train_indices) / len(dataframe),
        "validation_fraction": len(validation_indices) / len(dataframe),
        "n_examples": len(dataframe),
        "n_parent_proteins": int(dataframe["parent_protein_id"].nunique()),
        "threshold": config.PREDICTION_THRESHOLD,
        "train_class_counts": pd.Series(labels[train_indices]).value_counts().sort_index().to_dict(),
        "validation_class_counts": pd.Series(labels[validation_indices]).value_counts().sort_index().to_dict(),
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "training_validation_metrics_plot": comparison_plot,
        "evaluation_plots": evaluation_plots,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    with (config.OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nValidation metrics:")
    for metric, value in validation_metrics.items():
        print(f"  {metric}: {value}")
    print(
        "\nSaved train-only fitted model, validation predictions, and metrics to: "
        f"{config.OUTPUT_DIR}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a classical ML B-cell epitope classifier."
    )
    parser.add_argument("--dataset", choices=("bcell", "sars"))
    parser.add_argument(
        "--model",
        choices=("logistic_regression", "random_forest", "extra_trees", "svm"),
    )
    parser.add_argument("--features", choices=("aac", "aac_dipeptide"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CFG()
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.model:
        config.MODEL_TYPE = args.model
    if args.features:
        config.FEATURE_SET = args.features
    config.VALIDATION_SIZE = args.validation_size
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    run_training(config)


if __name__ == "__main__":
    main()