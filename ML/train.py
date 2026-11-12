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


def _build_configured_features(dataframe: pd.DataFrame, config: CFG) -> pd.DataFrame:
    return build_feature_frame(
        dataframe,
        use_peptide_features=config.USE_PEPTIDE_FEATURES,
        use_protein_features=config.USE_PROTEIN_FEATURES,
        use_biological_features=config.USE_BIOLOGICAL_FEATURES,
        peptide_feature_type=config.PEPTIDE_FEATURE_TYPE,
        feature_set=config.FEATURE_SET,
        use_aac_features=config.EFFECTIVE_AAC_FEATURES,
        use_dpc_features=config.EFFECTIVE_DPC_FEATURES,
    )


def _select_external_model_from_bcell(
    config: CFG,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    """Select model parameters, compute internal 5-fold CV metrics, and choose decision threshold on B-cell."""
    folds, _ = make_cross_validation_splits(
        labels,
        groups,
        dataset_name="bcell",
        n_splits=5,
        random_state=config.RANDOM_STATE,
    )
    base_parameters = _configured_model_parameters(config)
    overrides = (
        config.HYPERPARAMETER_TUNING_CANDIDATES.get(config.MODEL_TYPE, [])
        if config.HYPERPARAMETER_TUNING
        else []
    )
    candidates = [base_parameters]
    candidates.extend(
        {**base_parameters, **override} for override in overrides
    )

    candidate_results = []
    for candidate_id, parameters in enumerate(candidates):
        candidate_config = copy.copy(config)
        for key, value in parameters.items():
            setattr(candidate_config, key, value)
        candidate_config.HYPERPARAMETER_TUNING = False
        candidate_config.validate()

        oof_scores = np.full(len(labels), np.nan, dtype=float)
        fold_metrics_list = []
        for fold_number, (train_indices, validation_indices) in enumerate(folds, 1):
            fit_features, fit_labels, _ = smote_resample(
                features.iloc[train_indices],
                labels[train_indices],
                enabled=config.USE_SMOTE,
                random_state=config.RANDOM_STATE,
                k_neighbors=config.SMOTE_K_NEIGHBORS,
            )
            model = _build_configured_model(candidate_config)
            model.fit(fit_features, fit_labels)
            val_scores = model.predict_proba(
                features.iloc[validation_indices]
            )[:, 1]
            oof_scores[validation_indices] = val_scores
            fold_metrics = calculate_metrics(
                labels[validation_indices],
                val_scores,
                threshold=config.PREDICTION_THRESHOLD,
            )
            fold_metrics_list.append({"fold": fold_number, **fold_metrics})

        oof_metrics = calculate_metrics(
            labels,
            oof_scores,
            threshold=config.PREDICTION_THRESHOLD,
        )
        numeric_metric_names = [
            key
            for key, val in fold_metrics_list[0].items()
            if isinstance(val, (int, float)) and key != "fold"
        ]
        mean_fold_metrics = {
            metric: float(np.mean([f[metric] for f in fold_metrics_list]))
            for metric in numeric_metric_names
        }
        std_fold_metrics = {
            metric: float(np.std([f[metric] for f in fold_metrics_list]))
            for metric in numeric_metric_names
        }
        selection_metric = config.HYPERPARAMETER_TUNING_METRIC
        score = float(oof_metrics[selection_metric])
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "parameters": parameters,
                "selection_metric": selection_metric,
                "selection_score": score,
                "oof_scores": oof_scores,
                "oof_metrics": oof_metrics,
                "fold_metrics": fold_metrics_list,
                "mean_fold_metrics": mean_fold_metrics,
                "std_fold_metrics": std_fold_metrics,
            }
        )

    selected = max(candidate_results, key=lambda result: result["selection_score"])
    threshold_values = np.unique(np.r_[0.0, selected["oof_scores"], 1.0])
    threshold_mcc = [
        matthews_corrcoef(labels, selected["oof_scores"] >= value)
        for value in threshold_values
    ]
    best_mcc = max(threshold_mcc)
    tied_thresholds = [
        float(value)
        for value, score in zip(threshold_values, threshold_mcc)
        if np.isclose(score, best_mcc)
    ]
    selected_threshold = min(tied_thresholds, key=lambda value: abs(value - 0.5))

    oof_metrics_at_selected = calculate_metrics(
        labels,
        selected["oof_scores"],
        threshold=selected_threshold,
    )
    fold_metrics_selected_threshold = []
    for fold_report in selected["fold_metrics"]:
        fold_idx = fold_report["fold"]
        val_idx = folds[fold_idx - 1][1]
        val_fold_metrics = calculate_metrics(
            labels[val_idx],
            selected["oof_scores"][val_idx],
            threshold=selected_threshold,
        )
        fold_metrics_selected_threshold.append({"fold": fold_idx, **val_fold_metrics})
    mean_fold_metrics_selected = {
        metric: float(np.mean([f[metric] for f in fold_metrics_selected_threshold]))
        for metric in numeric_metric_names
    }
    std_fold_metrics_selected = {
        metric: float(np.std([f[metric] for f in fold_metrics_selected_threshold]))
        for metric in numeric_metric_names
    }

    comparison = [
        {
            "candidate_id": result["candidate_id"],
            "parameters": result["parameters"],
            "selection_metric": result["selection_metric"],
            "selection_score": result["selection_score"],
            "selected": result is selected,
            **{
                f"oof_{metric}": value
                for metric, value in result["oof_metrics"].items()
            },
        }
        for result in candidate_results
    ]
    selection = {
        "parameters": selected["parameters"],
        "selection_metric": selected["selection_metric"],
        "selection_score": selected["selection_score"],
        "oof_metrics_at_configured_threshold": selected["oof_metrics"],
        "oof_metrics_at_selected_threshold": oof_metrics_at_selected,
        "selected_threshold_from_bcell_oof_mcc": selected_threshold,
        "threshold_selection_mcc": float(best_mcc),
        "candidate_count": len(candidates),
        "fold_metrics": selected["fold_metrics"],
        "mean_fold_metrics": selected["mean_fold_metrics"],
        "std_fold_metrics": selected["std_fold_metrics"],
        "fold_metrics_at_selected_threshold": fold_metrics_selected_threshold,
        "mean_fold_metrics_at_selected_threshold": mean_fold_metrics_selected,
        "std_fold_metrics_at_selected_threshold": std_fold_metrics_selected,
    }
    return selection, selected["oof_scores"], comparison


def _run_external_sars(config: CFG) -> dict[str, object]:
    """Fit/choose exclusively on B-cell, then evaluate all SARS rows once."""
    bcell = load_labeled_dataset(
        "bcell", config.DATA_DIR, max_peptide_length=config.MAX_PEPTIDE_LENGTH
    )
    sars = load_labeled_dataset(
        "sars",
        config.EVALUATION_DATA_DIR or config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
    )
    sars_sequences = set(sars["peptide_seq"])
    overlap_mask = bcell["peptide_seq"].isin(sars_sequences)
    overlap_sequences = set(bcell.loc[overlap_mask, "peptide_seq"])
    sars_labels_by_sequence = dict(zip(sars["peptide_seq"], sars["target"]))
    bcell_labels_by_sequence = dict(zip(bcell["peptide_seq"], bcell["target"]))
    same_label_overlap = sum(
        bcell_labels_by_sequence[sequence] == sars_labels_by_sequence[sequence]
        for sequence in overlap_sequences
    )
    conflicting_label_overlap = len(overlap_sequences) - same_label_overlap

    # Keep every SARS peptide for the final score, but remove identical peptide
    # sequences from model development to avoid exact-sequence leakage.
    development = bcell.loc[~overlap_mask].reset_index(drop=True)
    labels = development["target"].to_numpy(dtype=int)
    groups = development["parent_protein_id"].astype(str).to_numpy()
    if np.unique(labels).size != 2:
        raise ValueError(
            "Removing B-cell/SARS sequence overlaps left only one class in the "
            "B-cell development set."
        )

    development_features = _build_configured_features(development, config)
    sars_features = _build_configured_features(sars, config)
    missing_features = development_features.columns.difference(sars_features.columns)
    extra_features = sars_features.columns.difference(development_features.columns)
    if len(missing_features) or len(extra_features):
        raise ValueError(
            "B-cell and SARS feature schemas differ after feature generation. "
            f"Missing in SARS: {missing_features.tolist()}; "
            f"extra in SARS: {extra_features.tolist()}."
        )
    sars_features = sars_features.reindex(columns=development_features.columns)

    selection, bcell_oof_scores, candidate_comparison = (
        _select_external_model_from_bcell(
            config, development_features, labels, groups
        )
    )
    final_config = copy.copy(config)
    for key, value in selection["parameters"].items():
        setattr(final_config, key, value)
    final_config.HYPERPARAMETER_TUNING = False
    final_config.validate()

    fit_features, fit_labels, synthetic_count = smote_resample(
        development_features,
        labels,
        enabled=config.USE_SMOTE,
        random_state=config.RANDOM_STATE,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
    )
    final_model = _build_configured_model(final_config)
    final_model.fit(fit_features, fit_labels)
    sars_scores = final_model.predict_proba(sars_features)[:, 1]
    sars_labels = sars["target"].to_numpy(dtype=int)
    selected_threshold = float(selection["selected_threshold_from_bcell_oof_mcc"])

    output_dir = config.OUTPUT_DIR / config.FEATURE_SET / "external_sars"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "peptide_seq": development["peptide_seq"],
            "target": labels,
            "parent_protein_id": groups,
            "bcell_oof_probability": bcell_oof_scores,
        }
    ).to_csv(output_dir / "bcell_oof_predictions.csv", index=False)
    pd.DataFrame(selection["fold_metrics"]).to_csv(
        output_dir / "bcell_fold_metrics.csv", index=False
    )
    pd.DataFrame(candidate_comparison).assign(
        parameters=lambda frame: frame["parameters"].map(json.dumps)
    ).to_csv(output_dir / "bcell_model_selection.csv", index=False)
    pd.DataFrame(
        {
            "peptide_seq": sars["peptide_seq"].to_numpy(),
            "target": sars_labels,
            "parent_protein_id": sars["parent_protein_id"].to_numpy(),
            "probability_positive": sars_scores,
            "prediction_at_0_5": (sars_scores >= 0.5).astype(int),
            "prediction_at_bcell_selected_threshold": (
                sars_scores >= selected_threshold
            ).astype(int),
        }
    ).to_csv(output_dir / "external_sars_predictions.csv", index=False)

    metrics_at_05 = calculate_metrics(sars_labels, sars_scores, threshold=0.5)
    metrics_at_selected = calculate_metrics(
        sars_labels, sars_scores, threshold=selected_threshold
    )
    bcell_plots = save_evaluation_plots(
        labels,
        bcell_oof_scores,
        threshold=selected_threshold,
        output_dir=output_dir,
        prefix="bcell_oof_",
        title_prefix="B-cell 5-Fold Group CV",
    )
    sars_plots = save_evaluation_plots(
        sars_labels,
        sars_scores,
        threshold=selected_threshold,
        output_dir=output_dir,
        prefix="external_sars_",
        title_prefix="External SARS",
    )
    # Default plot filenames for compatibility
    save_evaluation_plots(
        sars_labels,
        sars_scores,
        threshold=selected_threshold,
        output_dir=output_dir,
        prefix="",
        title_prefix="External SARS",
    )
    overlap_report = {
        "unique_exact_peptide_sequences": len(overlap_sequences),
        "same_label": int(same_label_overlap),
        "opposite_label": int(conflicting_label_overlap),
        "bcell_rows_excluded_from_development": int(overlap_mask.sum()),
        "sars_rows_retained_for_external_evaluation": int(len(sars)),
    }
    report: dict[str, object] = {
        "training_mode": "external_sars",
        "training_dataset": "bcell",
        "external_evaluation_dataset": "sars",
        "external_evaluation_split": "all cleaned SARS rows evaluated once; no SARS split",
        "sars_labels_used_for_model_or_threshold_selection": False,
        "internal_selection_protocol": "5-fold StratifiedGroupKFold on B-cell, grouped by parent_protein_id",
        "primary_selection_metric": (
            "average_precision"
            if config.HYPERPARAMETER_TUNING_METRIC == "pr_auc"
            else config.HYPERPARAMETER_TUNING_METRIC
        ),
        "model": config.MODEL_TYPE,
        "feature_set": config.FEATURE_SET,
        "feature_columns": development_features.columns.tolist(),
        "bcell_rows_before_overlap_exclusion": int(len(bcell)),
        "bcell_rows_used_for_development": int(len(development)),
        "bcell_proteins_used_for_development": int(development.parent_protein_id.nunique()),
        "bcell_class_counts_used_for_development": pd.Series(labels).value_counts().sort_index().to_dict(),
        "external_sars_rows": int(len(sars)),
        "external_sars_proteins": int(sars.parent_protein_id.nunique()),
        "external_sars_class_counts": pd.Series(sars_labels).value_counts().sort_index().to_dict(),
        "exact_sequence_overlap_audit": overlap_report,
        "internal_bcell_cv": {
            "protocol": "5-fold StratifiedGroupKFold on B-cell, grouped by parent_protein_id",
            "folds": selection["fold_metrics"],
            "mean_metrics": selection["mean_fold_metrics"],
            "std_metrics": selection["std_fold_metrics"],
            "folds_at_selected_threshold": selection["fold_metrics_at_selected_threshold"],
            "mean_metrics_at_selected_threshold": selection["mean_fold_metrics_at_selected_threshold"],
            "std_metrics_at_selected_threshold": selection["std_fold_metrics_at_selected_threshold"],
            "oof_metrics_threshold_0_5": selection["oof_metrics_at_configured_threshold"],
            "oof_metrics_bcell_selected_threshold": selection["oof_metrics_at_selected_threshold"],
            "selected_threshold": selected_threshold,
            "threshold_selection_mcc": selection["threshold_selection_mcc"],
            "fold_metrics_path": str(output_dir / "bcell_fold_metrics.csv"),
            "predictions_path": str(output_dir / "bcell_oof_predictions.csv"),
            "evaluation_plots": bcell_plots,
        },
        "external_sars_evaluation": {
            "protocol": "One-shot external evaluation on complete cleaned SARS dataset",
            "rows": int(len(sars)),
            "proteins": int(sars.parent_protein_id.nunique()),
            "class_counts": pd.Series(sars_labels).value_counts().sort_index().to_dict(),
            "metrics_threshold_0_5": metrics_at_05,
            "metrics_bcell_selected_threshold": metrics_at_selected,
            "predictions_path": str(output_dir / "external_sars_predictions.csv"),
            "evaluation_plots": sars_plots,
        },
        "model_selection": selection,
        "resampling": {
            "method": "SMOTE" if config.USE_SMOTE else "none",
            "scope": "B-cell training folds and final B-cell refit only",
            "synthetic_rows_final_refit": int(synthetic_count),
        },
        "class_weight": config.CLASS_WEIGHT,
        "external_sars_metrics_threshold_0_5": metrics_at_05,
        "external_sars_metrics_bcell_selected_threshold": metrics_at_selected,
        "evaluation_plots": sars_plots,
        "predictions_path": str(output_dir / "external_sars_predictions.csv"),
        "artifact_directory": str(output_dir),
        "interpretation_limit": (
            "The cleaned SARS dataset contains one parent protein; this is a "
            "single-protein external evaluation, not broad unseen-protein validation."
        ),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    with (output_dir / "external_sars_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    bundle = {
        "model": final_model,
        "feature_names": development_features.columns.tolist(),
        "feature_profile": config.FEATURE_SET,
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_aac_features": config.EFFECTIVE_AAC_FEATURES,
        "use_dpc_features": config.EFFECTIVE_DPC_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "peptide_feature_type": config.PEPTIDE_FEATURE_TYPE,
        "prediction_threshold": selected_threshold,
        "training_dataset": "bcell",
        "external_evaluation_dataset": "sars",
        "selected_hyperparameters": selection["parameters"],
        "exact_overlap_sequences_excluded_from_training": len(overlap_sequences),
    }
    dump(bundle, output_dir / "peptide_classifier.joblib")

    print("\n" + "=" * 96)
    print("EXPERIMENT EVALUATION SUMMARY: B-cell 5-Fold Group CV vs External SARS")
    print("=" * 96)
    print(
        f"{'Metric':<16} | {'B-cell CV (Mean ± SD)':<24} | "
        f"{'B-cell Pooled OOF':<18} | {'SARS (Thresh 0.5)':<18} | "
        f"{'SARS (Thresh ' + f'{selected_threshold:.3f})':<20}"
    )
    print("-" * 96)
    summary_metrics = [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC (AP)"),
        ("accuracy", "Accuracy"),
        ("recall", "Recall / Sens."),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("f1", "F1-Score"),
        ("mcc", "MCC"),
    ]
    for metric_key, metric_name in summary_metrics:
        bcell_mean = selection["mean_fold_metrics"].get(metric_key, 0.0)
        bcell_std = selection["std_fold_metrics"].get(metric_key, 0.0)
        bcell_oof = selection["oof_metrics_at_configured_threshold"].get(metric_key, 0.0)
        sars_05 = metrics_at_05.get(metric_key, 0.0)
        sars_sel = metrics_at_selected.get(metric_key, 0.0)
        print(
            f"{metric_name:<16} | {bcell_mean:.4f} ± {bcell_std:.4f}          | "
            f"{bcell_oof:<18.4f} | {sars_05:<18.4f} | {sars_sel:<20.4f}"
        )
    print("=" * 96 + "\n")
    print(
        f"External SARS evaluation complete: {len(sars):,} peptides from "
        f"{sars.parent_protein_id.nunique():,} protein(s); "
        f"excluded {len(overlap_sequences):,} exact B-cell/SARS peptide overlaps from training."
    )
    print(f"Saved experiment artifacts to: {output_dir}")
    return report


def _run_training_single(
    config: CFG,
    supplied_split: tuple[np.ndarray, np.ndarray] | None = None,
    split_strategy_override: str | None = None,
) -> dict[str, object]:
    """Run one training fold using the existing dataset-specific pipeline."""
    config.validate()

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
        "use_aac_features": config.EFFECTIVE_AAC_FEATURES,
        "use_dpc_features": config.EFFECTIVE_DPC_FEATURES,
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
        "use_aac_features": config.EFFECTIVE_AAC_FEATURES,
        "use_dpc_features": config.EFFECTIVE_DPC_FEATURES,
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
    if config.TRAINING_MODE in {"external_sars", "bcell_to_sars"}:
        return _run_external_sars(config)

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
            profile_output_dir / "cross_validation" / f"fold_{fold_number}"
        )
        fold_config._CV_FOLD_NUMBER = fold_number
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
        choices=("same_dataset", "bcell_to_sars", "external_sars"),
        help=(
            "Use internal dataset CV or train/select on B-cell only and "
            "evaluate all cleaned SARS rows once. bcell_to_sars is a legacy alias."
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
        help="Legacy peptide-feature switch; use --aac-features and --dpc-features for independent groups.",
    )
    parser.add_argument(
        "--protein-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Independently enable or disable protein composition and position features.",
    )
    parser.add_argument(
        "--biological-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the eight biological descriptors.",
    )
    parser.add_argument(
        "--aac-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Independently enable or disable peptide AAC features.",
    )
    parser.add_argument(
        "--dpc-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Independently enable or disable dipeptide composition features.",
    )
    parser.add_argument(
        "--peptide-feature-type",
        choices=("aac", "aac_dipeptide"),
        help="Legacy composition setting used when independent AAC/DPC switches are omitted.",
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
    if args.aac_features is not None:
        config.USE_AAC_FEATURES = args.aac_features
    if args.dpc_features is not None:
        config.USE_DPC_FEATURES = args.dpc_features
    if args.peptide_feature_type:
        config.PEPTIDE_FEATURE_TYPE = args.peptide_feature_type
    config.VALIDATION_SIZE = args.validation_size
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    run_training(config)


if __name__ == "__main__":
    main()
