"""Model construction and selection routines for classical ML."""

import copy

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from ML.CFG import CFG
from ML.dataset import make_cross_validation_splits
from ML.evaluation import calculate_metrics
from ML.model import build_model
from ML.oversampling import smote_resample


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
