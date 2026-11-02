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
from ML.oversampling import smote_resample
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
        ("Balanced accuracy", "balanced_accuracy"),
        ("F1", "f1"),
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


def _build_configured_features(dataframe: pd.DataFrame, config: CFG) -> pd.DataFrame:
    return build_feature_frame(
        dataframe,
        use_peptide_features=config.USE_PEPTIDE_FEATURES,
        use_protein_features=config.USE_PROTEIN_FEATURES,
        use_biological_features=config.USE_BIOLOGICAL_FEATURES,
        peptide_feature_type=config.PEPTIDE_FEATURE_TYPE,
    )


def _run_bcell_to_sars(config: CFG) -> dict[str, object]:
    """Train on all cleaned B-cell rows and score a stratified SARS holdout."""
    bcell_data = load_labeled_dataset(
        dataset_name="bcell",
        data_dir=config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
    )
    sars_data = load_labeled_dataset(
        dataset_name="sars",
        data_dir=config.EVALUATION_DATA_DIR or config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
    )
    bcell_features = _build_configured_features(bcell_data, config)
    sars_features = _build_configured_features(sars_data, config)
    if not bcell_features.columns.equals(sars_features.columns):
        raise ValueError(
            "B-cell and SARS feature columns do not match. "
            "Check that both datasets contain the selected feature inputs."
        )

    bcell_labels = bcell_data["target"].to_numpy(dtype=int)
    sars_labels = sars_data["target"].to_numpy(dtype=int)
    _, sars_validation_indices = make_stratified_split(
        labels=sars_labels,
        validation_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
    )
    sars_train_indices = np.setdiff1d(
        np.arange(len(sars_data)),
        sars_validation_indices,
        assume_unique=True,
    )

    original_bcell_size = len(bcell_labels)
    fit_features, fit_labels, synthetic_count = smote_resample(
        bcell_features,
        bcell_labels,
        enabled=config.USE_SMOTE,
        random_state=config.RANDOM_STATE,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
    )
    if config.USE_SMOTE:
        print(
            "Cross-dataset mode: SMOTE was applied to B-cell training rows only; "
            "SARS validation rows remain untouched."
        )
    else:
        print("Cross-dataset mode: SMOTE disabled.")

    model = build_model(
        model_type=config.MODEL_TYPE,
        random_state=config.RANDOM_STATE,
        random_forest_trees=config.RANDOM_FOREST_TREES,
    )
    model.fit(fit_features, fit_labels)

    train_probabilities = model.predict_proba(bcell_features)[:, 1]
    train_metrics = calculate_metrics(
        bcell_labels,
        train_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
    )
    validation_features = sars_features.iloc[sars_validation_indices]
    validation_labels = sars_labels[sars_validation_indices]
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    validation_metrics = calculate_metrics(
        validation_labels,
        validation_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
    )

    output_dir = config.OUTPUT_DIR / "bcell_to_sars"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_plot = save_train_validation_metrics_plot(
        train_metrics,
        validation_metrics,
        output_dir,
        train_label="B-cell training",
        validation_label="SARS holdout",
        title="B-cell training vs SARS holdout",
    )
    evaluation_plots = save_evaluation_plots(
        validation_labels,
        validation_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
        output_dir=output_dir,
    )
    sars_validation = sars_data.iloc[sars_validation_indices]
    pd.DataFrame(
        {
            "peptide_seq": sars_validation["peptide_seq"].to_numpy(),
            "target": validation_labels,
            "parent_protein_id": sars_validation[
                "parent_protein_id"
            ].to_numpy(),
            "is_synthetic": False,
            "validation_probability": validation_probabilities,
            "validation_prediction": (
                validation_probabilities >= config.PREDICTION_THRESHOLD
            ).astype(int),
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)

    sars_sequences = set(sars_data["peptide_seq"])
    bcell_sequences = set(bcell_data["peptide_seq"])
    overlapping_sequences = len(sars_sequences.intersection(bcell_sequences))
    if overlapping_sequences:
        print(
            "Warning: "
            f"{overlapping_sequences} unique peptide sequences occur in both "
            "B-cell training data and cleaned SARS data."
        )

    bundle = {
        "model": model,
        "feature_names": bcell_features.columns.tolist(),
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "peptide_feature_type": config.PEPTIDE_FEATURE_TYPE,
        "prediction_threshold": config.PREDICTION_THRESHOLD,
        "training_dataset": "bcell",
        "evaluation_dataset": "sars",
        "training_mode": config.TRAINING_MODE,
        "artifact_directory": str(output_dir),
        "random_state": config.RANDOM_STATE,
        "use_smote": config.USE_SMOTE,
        "smote_k_neighbors": config.SMOTE_K_NEIGHBORS,
    }
    dump(bundle, output_dir / "peptide_classifier.joblib")

    report: dict[str, object] = {
        "training_dataset": "bcell",
        "evaluation_dataset": "sars",
        "training_mode": config.TRAINING_MODE,
        "model": config.MODEL_TYPE,
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "peptide_feature_type": config.PEPTIDE_FEATURE_TYPE,
        "split_strategy": "stratified_sars_holdout",
        "train_size": original_bcell_size,
        "validation_size": len(sars_validation_indices),
        "sars_train_unused_size": len(sars_train_indices),
        "train_fraction": 1.0,
        "validation_fraction": len(sars_validation_indices) / len(sars_data),
        "n_synthetic_examples": synthetic_count,
        "bcell_class_counts": (
            pd.Series(bcell_labels).value_counts().sort_index().to_dict()
        ),
        "sars_class_counts": (
            pd.Series(sars_labels).value_counts().sort_index().to_dict()
        ),
        "sars_train_class_counts": (
            pd.Series(sars_labels[sars_train_indices])
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "validation_class_counts": (
            pd.Series(validation_labels).value_counts().sort_index().to_dict()
        ),
        "overlapping_unique_peptide_sequences": overlapping_sequences,
        "threshold": config.PREDICTION_THRESHOLD,
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "training_validation_metrics_plot": comparison_plot,
        "evaluation_plots": evaluation_plots,
        "validation_predictions": str(output_dir / "validation_predictions.csv"),
        "artifact_directory": str(output_dir),
        "resampling": (
            {
                "method": "SMOTE",
                "scope": "B-cell training data only",
                "k_neighbors": config.SMOTE_K_NEIGHBORS,
            }
            if config.USE_SMOTE
            else {"method": "none"}
        ),
        "validation_metrics_interpretation": (
            "Metrics use the stratified SARS holdout only. The remaining "
            "SARS rows were not used for training or evaluation."
        ),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(
        f"Cross-dataset split: train on {original_bcell_size:,} B-cell rows; "
        f"evaluate on {len(sars_validation_indices):,}/{len(sars_data):,} "
        f"SARS rows ({config.VALIDATION_SIZE:.0%} holdout)."
    )
    print(
        f"SARS holdout ROC-AUC={validation_metrics['roc_auc']:.4f}, "
        f"PR-AUC={validation_metrics['pr_auc']:.4f}, "
        f"F1={validation_metrics['f1']:.4f}"
    )
    print(f"Saved cross-dataset artifacts to: {output_dir}")
    return report


def run_training(config: CFG) -> dict[str, object]:
    """Run same-dataset training or B-cell-to-SARS transfer evaluation."""
    config.validate()
    if config.TRAINING_MODE == "bcell_to_sars":
        return _run_bcell_to_sars(config)

    dataframe = load_labeled_dataset(
        dataset_name=config.DATASET_NAME,
        data_dir=config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
    )
    feature_frame = build_feature_frame(
        dataframe,
        use_peptide_features=config.USE_PEPTIDE_FEATURES,
        use_protein_features=config.USE_PROTEIN_FEATURES,
        use_biological_features=config.USE_BIOLOGICAL_FEATURES,
        peptide_feature_type=config.PEPTIDE_FEATURE_TYPE,
    )
    labels = dataframe["target"].to_numpy(dtype=int)
    resampled_metadata = dataframe[
        ["peptide_seq", "target", "parent_protein_id"]
    ].copy()
    original_size = len(labels)
    feature_frame, labels, synthetic_count = smote_resample(
        feature_frame,
        labels,
        enabled=config.USE_SMOTE,
        random_state=config.RANDOM_STATE,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
    )
    if config.USE_SMOTE:
        if synthetic_count:
            synthetic_metadata = pd.DataFrame(
                {
                    "peptide_seq": [""] * synthetic_count,
                    "target": labels[original_size:],
                    "parent_protein_id": [pd.NA] * synthetic_count,
                }
            )
            resampled_metadata = pd.concat(
                [resampled_metadata, synthetic_metadata],
                ignore_index=True,
            )
        print(
            "Pre-split SMOTE enabled: "
            f"{original_size:,} -> {len(labels):,} examples "
            f"({synthetic_count:,} synthetic). Validation is resampled and "
            "must not be interpreted as an unbiased generalization estimate."
        )
    else:
        print("Pre-split SMOTE disabled.")
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
        f"Train: {len(train_indices):,} ({len(train_indices) / len(labels):.0%}); "
        f"validation: {len(validation_indices):,} "
        f"({len(validation_indices) / len(labels):.0%})"
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
            "peptide_seq": resampled_metadata.iloc[validation_indices][
                "peptide_seq"
            ].fillna("").to_numpy(),
            "target": labels[validation_indices],
            "parent_protein_id": resampled_metadata.iloc[validation_indices][
                "parent_protein_id"
            ].to_numpy(),
            "is_synthetic": (
                resampled_metadata.iloc[validation_indices][
                    "parent_protein_id"
                ].isna().to_numpy()
            ),
            "validation_probability": validation_probabilities,
            "validation_prediction": (
                validation_probabilities >= config.PREDICTION_THRESHOLD
            ).astype(int),
        }
    ).to_csv(config.OUTPUT_DIR / "validation_predictions.csv", index=False)
    bundle = {
        "model": model,
        "feature_names": feature_frame.columns.tolist(),
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "peptide_feature_type": config.PEPTIDE_FEATURE_TYPE,
        "prediction_threshold": config.PREDICTION_THRESHOLD,
        "dataset_name": config.DATASET_NAME,
        "training_mode": config.TRAINING_MODE,
        "validation_size": config.VALIDATION_SIZE,
        "random_state": config.RANDOM_STATE,
        "use_smote": config.USE_SMOTE,
        "smote_k_neighbors": config.SMOTE_K_NEIGHBORS,
    }
    dump(bundle, config.OUTPUT_DIR / "peptide_classifier.joblib")

    report: dict[str, object] = {
        "dataset": config.DATASET_NAME,
        "training_mode": config.TRAINING_MODE,
        "model": config.MODEL_TYPE,
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "peptide_feature_type": config.PEPTIDE_FEATURE_TYPE,
        "split_strategy": (
            "smote_before_stratified_split"
            if config.USE_SMOTE
            else "stratified_by_label"
        ),
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "train_fraction": len(train_indices) / len(labels),
        "validation_fraction": len(validation_indices) / len(labels),
        "resampling": (
            {
                "method": "SMOTE",
                "scope": "before train/validation split",
                "k_neighbors": config.SMOTE_K_NEIGHBORS,
                "validation_leakage_warning": True,
            }
            if config.USE_SMOTE
            else {"method": "none"}
        ),
        "validation_metrics_interpretation": (
            "Biased: SMOTE was applied before the train/validation split."
            if config.USE_SMOTE
            else "Validation rows were not oversampled."
        ),
        "n_examples": len(dataframe),
        "n_examples_after_resampling": len(labels),
        "n_synthetic_examples": len(labels) - len(dataframe),
        "n_parent_proteins": int(dataframe["parent_protein_id"].nunique()),
        "threshold": config.PREDICTION_THRESHOLD,
        "train_class_counts": pd.Series(labels[train_indices]).value_counts().sort_index().to_dict(),
        "validation_class_counts": pd.Series(labels[validation_indices]).value_counts().sort_index().to_dict(),
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "training_validation_metrics_plot": comparison_plot,
        "evaluation_plots": evaluation_plots,
        "artifact_directory": str(config.OUTPUT_DIR),
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
        "\nSaved fitted model bundle, validation predictions, and metrics to: "
        f"{config.OUTPUT_DIR}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a classical ML B-cell epitope classifier."
    )
    parser.add_argument("--dataset", choices=("bcell", "sars"))
    parser.add_argument(
        "--training-mode",
        choices=("same_dataset", "bcell_to_sars"),
        help=(
            "Use the existing same-dataset split or train on all B-cell rows "
            "and evaluate on a stratified SARS holdout."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("logistic_regression", "random_forest", "extra_trees", "svm"),
    )
    parser.add_argument(
        "--peptide-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable peptide AAC features.",
    )
    parser.add_argument(
        "--protein-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable protein AAC and position features (requires peptide features).",
    )
    parser.add_argument(
        "--biological-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the eight biological descriptors.",
    )
    parser.add_argument(
        "--peptide-feature-type",
        choices=("aac", "aac_dipeptide"),
        help="Peptide feature composition when peptide features are enabled.",
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CFG()
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.training_mode:
        config.TRAINING_MODE = args.training_mode
    if args.model:
        config.MODEL_TYPE = args.model
    if args.peptide_features is not None:
        config.USE_PEPTIDE_FEATURES = args.peptide_features
    if args.protein_features is not None:
        config.USE_PROTEIN_FEATURES = args.protein_features
    if args.biological_features is not None:
        config.USE_BIOLOGICAL_FEATURES = args.biological_features
    if args.peptide_feature_type:
        config.PEPTIDE_FEATURE_TYPE = args.peptide_feature_type
    config.VALIDATION_SIZE = args.validation_size
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    run_training(config)


if __name__ == "__main__":
    main()