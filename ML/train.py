"""Train and evaluate a peptide classifier with a protein-group holdout."""

import argparse
import copy
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
from ML.dataset import (
    make_cross_validation_splits,
    load_labeled_dataset,
    make_grouped_split,
    make_stratified_split,
)
from ML.model import build_model
from ML.oversampling import smote_resample
from ML.pre_dataset import build_feature_frame


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


def _build_configured_model(config: CFG):
    return build_model(
        model_type=config.MODEL_TYPE,
        random_state=config.RANDOM_STATE,
        random_forest_trees=config.RANDOM_FOREST_TREES,
        tree_max_depth=config.TREE_MAX_DEPTH,
        tree_min_samples_leaf=config.TREE_MIN_SAMPLES_LEAF,
        tree_min_samples_split=config.TREE_MIN_SAMPLES_SPLIT,
        tree_max_features=config.TREE_MAX_FEATURES,
        class_weight=config.CLASS_WEIGHT,
        logistic_regression_c=config.LOGISTIC_REGRESSION_C,
        logistic_regression_max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
        svm_c=config.SVM_C,
        svm_gamma=config.SVM_GAMMA,
    )


def _configured_model_parameters(config: CFG) -> dict[str, object]:
    if config.MODEL_TYPE == "logistic_regression":
        return {
            "LOGISTIC_REGRESSION_C": config.LOGISTIC_REGRESSION_C,
            "LOGISTIC_REGRESSION_MAX_ITER": config.LOGISTIC_REGRESSION_MAX_ITER,
        }
    if config.MODEL_TYPE == "svm":
        return {"SVM_C": config.SVM_C, "SVM_GAMMA": config.SVM_GAMMA}
    return {
        "RANDOM_FOREST_TREES": config.RANDOM_FOREST_TREES,
        "TREE_MAX_DEPTH": config.TREE_MAX_DEPTH,
        "TREE_MIN_SAMPLES_LEAF": config.TREE_MIN_SAMPLES_LEAF,
        "TREE_MIN_SAMPLES_SPLIT": config.TREE_MIN_SAMPLES_SPLIT,
        "TREE_MAX_FEATURES": config.TREE_MAX_FEATURES,
    }


def _fit_and_score_candidates(
    config: CFG,
    fit_features: pd.DataFrame,
    fit_labels: np.ndarray,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    validation_features: pd.DataFrame,
    validation_labels: np.ndarray,
) -> tuple[
    object,
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
    dict[str, float | int],
    list[dict[str, object]],
    dict[str, object],
]:
    """Fit configured hyperparameter candidates and select by validation score."""
    base_parameters = _configured_model_parameters(config)
    candidate_overrides = (
        config.HYPERPARAMETER_TUNING_CANDIDATES[config.MODEL_TYPE]
        if config.HYPERPARAMETER_TUNING
        else []
    )
    parameter_sets = [base_parameters]
    parameter_sets.extend(
        {**base_parameters, **overrides}
        for overrides in candidate_overrides
    )
    candidate_results = []
    for candidate_index, parameters in enumerate(parameter_sets):
        candidate_config = copy.copy(config)
        for name, value in parameters.items():
            setattr(candidate_config, name, value)
        candidate_config.HYPERPARAMETER_TUNING = False
        candidate_config.validate()

        model = _build_configured_model(candidate_config)
        model.fit(fit_features, fit_labels)
        train_probabilities = model.predict_proba(train_features)[:, 1]
        validation_probabilities = model.predict_proba(validation_features)[:, 1]
        train_metrics = calculate_metrics(
            train_labels,
            train_probabilities,
            threshold=config.PREDICTION_THRESHOLD,
        )
        validation_metrics = calculate_metrics(
            validation_labels,
            validation_probabilities,
            threshold=config.PREDICTION_THRESHOLD,
        )
        score = float(
            validation_metrics[config.HYPERPARAMETER_TUNING_METRIC]
        )
        if not np.isfinite(score):
            raise ValueError(
                "Hyperparameter tuning produced a non-finite validation "
                f"{config.HYPERPARAMETER_TUNING_METRIC} for candidate "
                f"{candidate_index}."
            )
        candidate_results.append(
            {
                "candidate_index": candidate_index,
                "model": model,
                "train_probabilities": train_probabilities,
                "validation_probabilities": validation_probabilities,
                "training_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "hyperparameters": parameters,
                "selection_score": score,
            }
        )

    best_candidate = max(
        candidate_results,
        key=lambda result: result["selection_score"],
    )
    comparison = []
    for result in candidate_results:
        comparison.append(
            {
                "candidate_index": result["candidate_index"],
                "hyperparameters": result["hyperparameters"],
                "selection_metric": config.HYPERPARAMETER_TUNING_METRIC,
                "selection_score": result["selection_score"],
                "selected": result is best_candidate,
                **{
                    f"validation_{name}": value
                    for name, value in result["validation_metrics"].items()
                },
                **{
                    f"training_{name}": value
                    for name, value in result["training_metrics"].items()
                },
            }
        )

    return (
        best_candidate["model"],
        best_candidate["train_probabilities"],
        best_candidate["validation_probabilities"],
        best_candidate["training_metrics"],
        best_candidate["validation_metrics"],
        comparison if config.HYPERPARAMETER_TUNING else [],
        best_candidate["hyperparameters"],
    )


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


def _build_configured_features(dataframe: pd.DataFrame, config: CFG) -> pd.DataFrame:
    return build_feature_frame(
        dataframe,
        use_peptide_features=config.USE_PEPTIDE_FEATURES,
        use_protein_features=config.USE_PROTEIN_FEATURES,
        use_biological_features=config.USE_BIOLOGICAL_FEATURES,
        peptide_feature_type=config.PEPTIDE_FEATURE_TYPE,
        feature_set=config.FEATURE_SET,
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
    bcell_labels = bcell_data["target"].to_numpy(dtype=int)
    sars_labels = sars_data["target"].to_numpy(dtype=int)
    fold_validation_indices = getattr(
        config,
        "_CV_SARS_VALIDATION_INDICES",
        None,
    )
    if fold_validation_indices is None:
        sars_train_indices, sars_validation_indices = make_stratified_split(
            labels=sars_labels,
            validation_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_STATE,
        )
    else:
        sars_validation_indices = np.asarray(fold_validation_indices, dtype=int)
        sars_train_indices = np.setdiff1d(
            np.arange(len(sars_labels)),
            sars_validation_indices,
        )
    bcell_features = _build_configured_features(bcell_data, config)
    sars_features = _build_configured_features(sars_data, config)
    if not bcell_features.columns.equals(sars_features.columns):
        raise ValueError(
            "B-cell and SARS feature columns do not match. "
            "Check that both datasets contain the selected feature inputs."
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

    (
        model,
        train_probabilities,
        validation_probabilities,
        train_metrics,
        validation_metrics,
        hyperparameter_comparison,
        selected_hyperparameters,
    ) = _fit_and_score_candidates(
        config=config,
        fit_features=fit_features,
        fit_labels=fit_labels,
        train_features=bcell_features,
        train_labels=bcell_labels,
        validation_features=sars_features.iloc[sars_validation_indices],
        validation_labels=sars_labels[sars_validation_indices],
    )
    validation_features = sars_features.iloc[sars_validation_indices]
    validation_labels = sars_labels[sars_validation_indices]

    output_dir = (
        config.OUTPUT_DIR
        if getattr(config, "_CV_FOLD_NUMBER", None) is not None
        else config.OUTPUT_DIR / "bcell_to_sars"
    )
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
    if hyperparameter_comparison:
        pd.DataFrame(hyperparameter_comparison).assign(
            hyperparameters=lambda frame: frame["hyperparameters"].map(
                json.dumps
            )
        ).to_csv(output_dir / "hyperparameter_comparison.csv", index=False)

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
        "feature_profile": config.FEATURE_SET,
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
        "hyperparameter_tuning": config.HYPERPARAMETER_TUNING,
        "selected_hyperparameters": selected_hyperparameters,
    }
    dump(bundle, output_dir / "peptide_classifier.joblib")

    report: dict[str, object] = {
        "training_dataset": "bcell",
        "evaluation_dataset": "sars",
        "training_mode": config.TRAINING_MODE,
        "model": config.MODEL_TYPE,
        "feature_set": config.FEATURE_SET,
        "class_labels": {"0": "negative", "1": "positive"},
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "peptide_feature_type": config.PEPTIDE_FEATURE_TYPE,
        "split_strategy": (
            "stratified_peptide_5_fold"
            if fold_validation_indices is not None
            else "stratified_peptide_sars_holdout"
        ),
        "fold": getattr(config, "_CV_FOLD_NUMBER", None),
        "train_size": original_bcell_size,
        "validation_size": len(sars_validation_indices),
        "sars_train_unused_size": len(sars_train_indices),
        "train_fraction": 1.0,
        "validation_fraction": len(sars_validation_indices) / len(sars_data),
        "sars_train_protein_count": int(
            sars_data.iloc[sars_train_indices]["parent_protein_id"].nunique()
        ),
        "sars_validation_protein_count": int(
            sars_data.iloc[sars_validation_indices]["parent_protein_id"].nunique()
        ),
        "sars_train_validation_proteins_overlap": bool(
            set(sars_data.iloc[sars_train_indices]["parent_protein_id"])
            & set(sars_data.iloc[sars_validation_indices]["parent_protein_id"])
        ),
        "train_protein_count": int(bcell_data["parent_protein_id"].nunique()),
        "validation_protein_count": int(
            sars_data.iloc[sars_validation_indices]["parent_protein_id"].nunique()
        ),
        "n_synthetic_examples": synthetic_count,
        "bcell_class_counts": (
            pd.Series(bcell_labels).value_counts().sort_index().to_dict()
        ),
        "train_class_counts": (
            pd.Series(bcell_labels).value_counts().sort_index().to_dict()
        ),
        "fit_class_counts": (
            pd.Series(fit_labels).value_counts().sort_index().to_dict()
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
        "hyperparameter_tuning": config.HYPERPARAMETER_TUNING,
        "hyperparameter_tuning_metric": (
            config.HYPERPARAMETER_TUNING_METRIC
            if config.HYPERPARAMETER_TUNING
            else None
        ),
        "selected_hyperparameters": selected_hyperparameters,
        "hyperparameter_comparison": hyperparameter_comparison,
        "hyperparameter_comparison_path": (
            str(output_dir / "hyperparameter_comparison.csv")
            if hyperparameter_comparison
            else None
        ),
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
            "Metrics use a stratified peptide-level SARS holdout. The remaining "
            "SARS rows were not used for fitting, but the same source protein "
            "may occur in both partitions."
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
        "stratified SARS peptide rows (peptide-level holdout; not a "
        "protein-level holdout)."
    )
    print(
        f"SARS holdout ROC-AUC={validation_metrics['roc_auc']:.4f}, "
        f"PR-AUC={validation_metrics['pr_auc']:.4f}, "
        f"Accuracy={validation_metrics['accuracy']:.4f}, "
        f"Recall={validation_metrics['recall']:.4f}, "
        f"Precision={validation_metrics['precision']:.4f}, "
        f"F1={validation_metrics['f1']:.4f}, "
        f"TN={validation_metrics['tn']}, FP={validation_metrics['fp']}, "
        f"FN={validation_metrics['fn']}, TP={validation_metrics['tp']}, "
        f"Balanced accuracy={validation_metrics['balanced_accuracy']:.4f}, "
        f"MCC={validation_metrics['mcc']:.4f}"
    )
    print(f"Saved cross-dataset artifacts to: {output_dir}")
    return report


def _run_training_single(
    config: CFG,
    supplied_split: tuple[np.ndarray, np.ndarray] | None = None,
    split_strategy_override: str | None = None,
) -> dict[str, object]:
    """Run one training fold using the existing dataset-specific pipeline."""
    config.validate()
    if config.TRAINING_MODE == "bcell_to_sars":
        return _run_bcell_to_sars(config)

    dataframe = load_labeled_dataset(
        dataset_name=config.DATASET_NAME,
        data_dir=config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
    )
    feature_frame = _build_configured_features(dataframe, config)
    labels = dataframe["target"].to_numpy(dtype=int)
    if supplied_split is not None:
        train_indices, validation_indices = supplied_split
        split_strategy = split_strategy_override or "cross_validation"
    elif config.DATASET_NAME.lower() == "sars":
        train_indices, validation_indices = make_stratified_split(
            labels=labels,
            validation_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_STATE,
        )
        split_strategy = "stratified_peptide_holdout"
    else:
        train_indices, validation_indices = make_grouped_split(
            labels=labels,
            groups=dataframe["parent_protein_id"].to_numpy(),
            validation_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_STATE,
        )
        split_strategy = "protein_group_holdout"
    train_features = feature_frame.iloc[train_indices]
    train_labels = labels[train_indices]
    original_size = len(labels)
    fit_features, fit_labels, synthetic_count = smote_resample(
        train_features,
        train_labels,
        enabled=config.USE_SMOTE,
        random_state=config.RANDOM_STATE,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
    )
    if config.USE_SMOTE:
        print(
            "Training-only SMOTE enabled: "
            f"{len(train_labels):,} -> {len(fit_labels):,} training examples "
            f"({synthetic_count:,} synthetic); validation "
            "remains untouched."
        )
    else:
        print("SMOTE disabled.")
    (
        model,
        train_probabilities,
        validation_probabilities,
        train_metrics,
        validation_metrics,
        hyperparameter_comparison,
        selected_hyperparameters,
    ) = _fit_and_score_candidates(
        config=config,
        fit_features=fit_features,
        fit_labels=fit_labels,
        train_features=train_features,
        train_labels=train_labels,
        validation_features=feature_frame.iloc[validation_indices],
        validation_labels=labels[validation_indices],
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
    split_description = (
        "Peptide-stratified"
        if split_strategy == "stratified_peptide_holdout"
        else "Protein-group"
    )
    print(
        f"{split_description} split: train={len(train_indices):,} "
        f"({len(train_indices) / original_size:.1%}, "
        f"{dataframe.iloc[train_indices]['parent_protein_id'].nunique():,} "
        f"proteins); "
        f"validation: {len(validation_indices):,} "
        f"({len(validation_indices) / original_size:.1%}, "
        f"{dataframe.iloc[validation_indices]['parent_protein_id'].nunique():,} "
        "proteins)"
    )
    print(
        f"Training ROC-AUC={train_metrics['roc_auc']:.4f}, "
        f"PR-AUC={train_metrics['pr_auc']:.4f}, "
        f"Accuracy={train_metrics['accuracy']:.4f}, "
        f"Recall={train_metrics['recall']:.4f}, "
        f"Precision={train_metrics['precision']:.4f}, "
        f"F1={train_metrics['f1']:.4f}, "
        f"TN={train_metrics['tn']}, FP={train_metrics['fp']}, "
        f"FN={train_metrics['fn']}, TP={train_metrics['tp']}, "
        f"Balanced accuracy={train_metrics['balanced_accuracy']:.4f}, "
        f"MCC={train_metrics['mcc']:.4f}"
    )
    print(
        f"Validation ROC-AUC={validation_metrics['roc_auc']:.4f}, "
        f"PR-AUC={validation_metrics['pr_auc']:.4f}, "
        f"Accuracy={validation_metrics['accuracy']:.4f}, "
        f"Recall={validation_metrics['recall']:.4f}, "
        f"Precision={validation_metrics['precision']:.4f}, "
        f"F1={validation_metrics['f1']:.4f}"
        f", TN={validation_metrics['tn']}, FP={validation_metrics['fp']}, "
        f"FN={validation_metrics['fn']}, TP={validation_metrics['tp']}, "
        f"Balanced accuracy={validation_metrics['balanced_accuracy']:.4f}, "
        f"MCC={validation_metrics['mcc']:.4f}"
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "peptide_seq": dataframe.iloc[validation_indices][
                "peptide_seq"
            ].to_numpy(),
            "target": labels[validation_indices],
            "parent_protein_id": dataframe.iloc[validation_indices][
                "parent_protein_id"
            ].to_numpy(),
            "is_synthetic": False,
            "validation_probability": validation_probabilities,
            "validation_prediction": (
                validation_probabilities >= config.PREDICTION_THRESHOLD
            ).astype(int),
        }
    ).to_csv(config.OUTPUT_DIR / "validation_predictions.csv", index=False)
    if hyperparameter_comparison:
        comparison_frame = pd.DataFrame(hyperparameter_comparison)
        comparison_frame["hyperparameters"] = comparison_frame[
            "hyperparameters"
        ].map(json.dumps)
        comparison_frame.to_csv(
            config.OUTPUT_DIR / "hyperparameter_comparison.csv",
            index=False,
        )
    bundle = {
        "model": model,
        "feature_names": feature_frame.columns.tolist(),
        "feature_profile": config.FEATURE_SET,
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
        "hyperparameter_tuning": config.HYPERPARAMETER_TUNING,
        "selected_hyperparameters": selected_hyperparameters,
    }
    dump(bundle, config.OUTPUT_DIR / "peptide_classifier.joblib")

    report: dict[str, object] = {
        "dataset": config.DATASET_NAME,
        "training_mode": config.TRAINING_MODE,
        "model": config.MODEL_TYPE,
        "feature_set": config.FEATURE_SET,
        "class_labels": {"0": "negative", "1": "positive"},
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "peptide_feature_type": config.PEPTIDE_FEATURE_TYPE,
        "split_strategy": split_strategy,
        "train_size": len(train_indices),
        "validation_size": len(validation_indices),
        "train_fraction": len(train_indices) / len(labels),
        "validation_fraction": len(validation_indices) / len(labels),
        "train_protein_count": int(
            dataframe.iloc[train_indices]["parent_protein_id"].nunique()
        ),
        "validation_protein_count": int(
            dataframe.iloc[validation_indices]["parent_protein_id"].nunique()
        ),
        "train_validation_proteins_overlap": bool(
            set(dataframe.iloc[train_indices]["parent_protein_id"])
            & set(dataframe.iloc[validation_indices]["parent_protein_id"])
        ),
        "resampling": (
            {
                "method": "SMOTE",
                "scope": "training proteins only",
                "k_neighbors": config.SMOTE_K_NEIGHBORS,
            }
            if config.USE_SMOTE
            else {"method": "none"}
        ),
        "validation_metrics_interpretation": (
            (
                "Validation is a stratified peptide-level holdout; the same "
                "source protein may occur in both partitions. SMOTE, when "
                "enabled, is applied only to training rows."
                if split_strategy == "stratified_peptide_holdout"
                else (
                    "Validation contains only original rows from proteins "
                    "excluded from training; SMOTE, when enabled, is applied "
                    "only to training rows."
                )
            )
        ),
        "n_examples": len(dataframe),
        "n_examples_after_resampling": len(fit_labels),
        "n_synthetic_examples": synthetic_count,
        "n_parent_proteins": int(dataframe["parent_protein_id"].nunique()),
        "threshold": config.PREDICTION_THRESHOLD,
        "train_class_counts": pd.Series(train_labels).value_counts().sort_index().to_dict(),
        "fit_class_counts": pd.Series(fit_labels).value_counts().sort_index().to_dict(),
        "validation_class_counts": pd.Series(labels[validation_indices]).value_counts().sort_index().to_dict(),
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "hyperparameter_tuning": config.HYPERPARAMETER_TUNING,
        "hyperparameter_tuning_metric": (
            config.HYPERPARAMETER_TUNING_METRIC
            if config.HYPERPARAMETER_TUNING
            else None
        ),
        "selected_hyperparameters": selected_hyperparameters,
        "hyperparameter_comparison": hyperparameter_comparison,
        "hyperparameter_comparison_path": (
            str(config.OUTPUT_DIR / "hyperparameter_comparison.csv")
            if hyperparameter_comparison
            else None
        ),
        "training_validation_metrics_plot": comparison_plot,
        "evaluation_plots": evaluation_plots,
        "validation_predictions": str(
            config.OUTPUT_DIR / "validation_predictions.csv"
        ),
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


def run_training(config: CFG) -> dict[str, object]:
    """Run five-fold CV, grouping proteins except for peptide-level SARS folds."""
    config.validate()
    config.OUTPUT_DIR = Path(config.OUTPUT_DIR)
    if config.TRAINING_MODE == "bcell_to_sars":
        sars_data = load_labeled_dataset(
            dataset_name="sars",
            data_dir=config.EVALUATION_DATA_DIR or config.DATA_DIR,
            max_peptide_length=config.MAX_PEPTIDE_LENGTH,
        )
        folds, split_strategy = make_cross_validation_splits(
            sars_data["target"].to_numpy(dtype=int),
            sars_data["parent_protein_id"].to_numpy(),
            dataset_name="sars",
            n_splits=5,
            random_state=config.RANDOM_STATE,
        )
        profile_output_dir = config.OUTPUT_DIR / config.FEATURE_SET
        summary_dir = profile_output_dir / "bcell_to_sars" / "cross_validation"
    else:
        dataframe = load_labeled_dataset(
            dataset_name=config.DATASET_NAME,
            data_dir=config.DATA_DIR,
            max_peptide_length=config.MAX_PEPTIDE_LENGTH,
        )
        folds, split_strategy = make_cross_validation_splits(
            dataframe["target"].to_numpy(dtype=int),
            dataframe["parent_protein_id"].to_numpy(),
            dataset_name=config.DATASET_NAME,
            n_splits=5,
            random_state=config.RANDOM_STATE,
        )
        profile_output_dir = config.OUTPUT_DIR / config.FEATURE_SET
        summary_dir = profile_output_dir / "cross_validation"

    fold_reports = []
    for fold_number, fold_split in enumerate(folds, 1):
        fold_config = copy.copy(config)
        fold_config.OUTPUT_DIR = (
            summary_dir.parent / f"fold_{fold_number}"
            if config.TRAINING_MODE == "bcell_to_sars"
            else profile_output_dir / "cross_validation" / f"fold_{fold_number}"
        )
        fold_config._CV_FOLD_NUMBER = fold_number
        if config.TRAINING_MODE == "bcell_to_sars":
            fold_config._CV_SARS_VALIDATION_INDICES = fold_split[1]
            fold_report = _run_training_single(fold_config)
        else:
            fold_report = _run_training_single(
                fold_config,
                supplied_split=fold_split,
                split_strategy_override=split_strategy,
            )
        fold_reports.append(fold_report)

    prediction_frames = []
    hyperparameter_comparison_rows = []
    for fold_number, fold_report in enumerate(fold_reports, 1):
        prediction_path = Path(fold_report["validation_predictions"])
        fold_predictions = pd.read_csv(prediction_path)
        probability_column = (
            "validation_probability"
            if "validation_probability" in fold_predictions
            else "probability"
        )
        fold_predictions["fold"] = fold_number
        fold_predictions["validation_probability"] = fold_predictions[
            probability_column
        ]
        prediction_frames.append(fold_predictions)
        for candidate in fold_report.get("hyperparameter_comparison", []):
            hyperparameter_comparison_rows.append(
                {"fold": fold_number, **candidate}
            )

    out_of_fold_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = summary_dir / "out_of_fold_predictions.csv"
    out_of_fold_predictions.to_csv(predictions_path, index=False)
    hyperparameter_comparison_path = None
    if config.HYPERPARAMETER_TUNING:
        hyperparameter_comparison_path = (
            summary_dir / "hyperparameter_comparison.csv"
        )
        comparison_frame = pd.DataFrame(hyperparameter_comparison_rows)
        comparison_frame["hyperparameters"] = comparison_frame[
            "hyperparameters"
        ].map(json.dumps)
        comparison_frame.to_csv(hyperparameter_comparison_path, index=False)
    pooled_metrics = calculate_metrics(
        out_of_fold_predictions["target"].to_numpy(dtype=int),
        out_of_fold_predictions["validation_probability"].to_numpy(dtype=float),
        threshold=config.PREDICTION_THRESHOLD,
    )
    validation_metric_names = fold_reports[0]["validation_metrics"]
    mean_validation_metrics = {
        metric: float(
            np.mean([report["validation_metrics"][metric] for report in fold_reports])
        )
        for metric in validation_metric_names
    }
    std_validation_metrics = {
        metric: float(
            np.std([report["validation_metrics"][metric] for report in fold_reports])
        )
        for metric in validation_metric_names
    }
    training_metric_names = fold_reports[0]["training_metrics"]
    mean_training_metrics = {
        metric: float(
            np.mean([report["training_metrics"][metric] for report in fold_reports])
        )
        for metric in training_metric_names
    }
    std_training_metrics = {
        metric: float(
            np.std([report["training_metrics"][metric] for report in fold_reports])
        )
        for metric in training_metric_names
    }
    report: dict[str, object] = {
        "dataset": config.DATASET_NAME,
        "training_mode": config.TRAINING_MODE,
        "model": config.MODEL_TYPE,
        "feature_set": config.FEATURE_SET,
        "class_labels": {"0": "negative", "1": "positive"},
        "split_strategy": split_strategy,
        "n_splits": 5,
        "fold_metrics": [
            {
                "fold": fold_number,
                "train_size": fold_report["train_size"],
                "validation_size": fold_report["validation_size"],
                "training_metrics": fold_report["training_metrics"],
                "validation_metrics": fold_report["validation_metrics"],
                "selected_hyperparameters": fold_report[
                    "selected_hyperparameters"
                ],
                "hyperparameter_comparison": fold_report[
                    "hyperparameter_comparison"
                ],
                "artifact_directory": fold_report["artifact_directory"],
            }
            for fold_number, fold_report in enumerate(fold_reports, 1)
        ],
        "mean_training_metrics": mean_training_metrics,
        "std_training_metrics": std_training_metrics,
        "mean_validation_metrics": mean_validation_metrics,
        "std_validation_metrics": std_validation_metrics,
        "training_metrics": mean_training_metrics,
        "validation_metrics": pooled_metrics,
        "hyperparameter_tuning": config.HYPERPARAMETER_TUNING,
        "hyperparameter_tuning_metric": (
            config.HYPERPARAMETER_TUNING_METRIC
            if config.HYPERPARAMETER_TUNING
            else None
        ),
        "hyperparameter_comparison": hyperparameter_comparison_rows,
        "hyperparameter_comparison_path": (
            str(hyperparameter_comparison_path)
            if hyperparameter_comparison_path is not None
            else None
        ),
        "hyperparameter_tuning_warning": (
            "Hyperparameters were selected using the same validation fold "
            "reported for evaluation. Fold metrics and pooled out-of-fold "
            "metrics may therefore be optimistic."
            if config.HYPERPARAMETER_TUNING
            else None
        ),
        "out_of_fold_predictions": str(predictions_path),
        "artifact_directory": str(summary_dir),
        "fold_reports": fold_reports,
        "validation_metrics_interpretation": (
            "Pooled out-of-fold predictions from five folds. B-cell folds hold "
            "out complete proteins; SARS folds stratify peptides and share the "
            "single source protein across folds."
            if config.DATASET_NAME.lower() == "sars"
            else "Pooled out-of-fold predictions from five stratified "
            "protein-group folds."
        ),
    }
    with (summary_dir / "cross_validation_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)
    print(
        f"5-fold CV complete ({split_strategy}). "
        f"Pooled ROC-AUC={pooled_metrics['roc_auc']:.4f}, "
        f"PR-AUC={pooled_metrics['pr_auc']:.4f}."
    )
    print(f"Saved cross-validation report to: {summary_dir}")
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
        "--feature-set",
        choices=("perbandingan_1", "proposed"),
        help=(
            "Select length-only peptide/protein features from the paper, or "
            "the current proposed feature engineering."
        ),
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
    if args.feature_set:
        config.FEATURE_SET = args.feature_set
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
