"""Model construction and selection routines for classical ML."""

import copy

import numpy as np
import pandas as pd

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
        xgboost_estimators=config.XGBOOST_ESTIMATORS,
        xgboost_max_depth=config.XGBOOST_MAX_DEPTH,
        xgboost_learning_rate=config.XGBOOST_LEARNING_RATE,
        xgboost_subsample=config.XGBOOST_SUBSAMPLE,
        xgboost_colsample_bytree=config.XGBOOST_COLSAMPLE_BYTREE,
        xgboost_scale_pos_weight=config.XGBOOST_SCALE_POS_WEIGHT,
        feature_set=config.FEATURE_SET,
    )


def _configured_model_parameters(config: CFG) -> dict[str, object]:
    if config.MODEL_TYPE == "logistic_regression":
        return {
            "LOGISTIC_REGRESSION_C": config.LOGISTIC_REGRESSION_C,
            "LOGISTIC_REGRESSION_MAX_ITER": config.LOGISTIC_REGRESSION_MAX_ITER,
        }
    if config.MODEL_TYPE == "svm":
        return {"SVM_C": config.SVM_C, "SVM_GAMMA": config.SVM_GAMMA}
    if config.MODEL_TYPE == "xgboost":
        return {
            "XGBOOST_ESTIMATORS": config.XGBOOST_ESTIMATORS,
            "XGBOOST_MAX_DEPTH": config.XGBOOST_MAX_DEPTH,
            "XGBOOST_LEARNING_RATE": config.XGBOOST_LEARNING_RATE,
            "XGBOOST_SUBSAMPLE": config.XGBOOST_SUBSAMPLE,
            "XGBOOST_COLSAMPLE_BYTREE": config.XGBOOST_COLSAMPLE_BYTREE,
            "XGBOOST_SCALE_POS_WEIGHT": config.XGBOOST_SCALE_POS_WEIGHT,
        }
    return {
        "RANDOM_FOREST_TREES": config.RANDOM_FOREST_TREES,
        "TREE_MAX_DEPTH": config.TREE_MAX_DEPTH,
        "TREE_MIN_SAMPLES_LEAF": config.TREE_MIN_SAMPLES_LEAF,
        "TREE_MIN_SAMPLES_SPLIT": config.TREE_MIN_SAMPLES_SPLIT,
        "TREE_MAX_FEATURES": config.TREE_MAX_FEATURES,
    }


def _print_tuning_results(
    title: str,
    candidate_results: list[dict[str, object]],
    *,
    index_key: str,
    parameters_key: str,
    metric_key: str,
    score_key: str,
    selected_result: dict[str, object],
) -> None:
    """Print the score and parameters for every evaluated candidate."""
    print(f"\n{title}")
    print("Candidate | Selection metric | Score | Selected | Parameters")
    for result in candidate_results:
        selected = result is selected_result
        print(
            f"{result[index_key]} | {result[metric_key]} | "
            f"{float(result[score_key]):.6f} | "
            f"{'yes' if selected else 'no'} | {result[parameters_key]}"
        )


def _select_hyperparameters_on_validation(
    config: CFG,
    fit_features: pd.DataFrame,
    fit_labels: np.ndarray,
    validation_features: pd.DataFrame,
    validation_labels: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select hyperparameters by scoring candidates on the outer validation fold."""
    base_parameters = _configured_model_parameters(config)
    if not config.HYPERPARAMETER_TUNING:
        return base_parameters, []

    parameter_sets = [base_parameters]
    parameter_sets.extend(
        {**base_parameters, **overrides}
        for overrides in config.HYPERPARAMETER_TUNING_CANDIDATES[
            config.MODEL_TYPE
        ]
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
        validation_scores = model.predict_proba(validation_features)[:, 1]
        validation_metrics = calculate_metrics(
            validation_labels,
            validation_scores,
            threshold=config.PREDICTION_THRESHOLD,
        )
        score = float(validation_metrics[config.HYPERPARAMETER_TUNING_METRIC])
        if not np.isfinite(score):
            raise ValueError(
                "Outer-validation hyperparameter selection produced a non-finite "
                f"{config.HYPERPARAMETER_TUNING_METRIC} for candidate "
                f"{candidate_index}."
            )
        candidate_results.append(
            {
                "candidate_index": candidate_index,
                "hyperparameters": parameters,
                "selection_metric": config.HYPERPARAMETER_TUNING_METRIC,
                "selection_score": score,
                "validation_metrics": validation_metrics,
            }
        )

    best_candidate = max(
        candidate_results,
        key=lambda result: result["selection_score"],
    )
    fold_number = getattr(config, "_CV_FOLD_NUMBER", None)
    tuning_scope = (
        f"Hyperparameter selection on outer validation fold {fold_number}"
        if fold_number is not None
        else "Hyperparameter selection on validation fold"
    )
    _print_tuning_results(
        tuning_scope,
        candidate_results,
        index_key="candidate_index",
        parameters_key="hyperparameters",
        metric_key="selection_metric",
        score_key="selection_score",
        selected_result=best_candidate,
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
            }
        )

    return best_candidate["hyperparameters"], comparison


def _select_external_model_from_bcell(
    config: CFG,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    """Select parameters and return their pooled B-cell OOF probabilities."""
    base_parameters = _configured_model_parameters(config)
    folds, _ = make_cross_validation_splits(
        labels,
        groups,
        dataset_name="bcell",
        n_splits=5,
        random_state=config.RANDOM_STATE,
    )
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
        for train_indices, validation_indices in folds:
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

        oof_metrics = calculate_metrics(
            labels,
            oof_scores,
            threshold=config.PREDICTION_THRESHOLD,
        )
        selection_metric = (
            config.HYPERPARAMETER_TUNING_METRIC
            if config.HYPERPARAMETER_TUNING
            else None
        )
        score = (
            float(oof_metrics[selection_metric])
            if selection_metric is not None
            else None
        )
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "parameters": parameters,
                "selection_metric": selection_metric,
                "selection_score": score,
                "oof_metrics": oof_metrics,
                "oof_scores": oof_scores,
            }
        )

    if config.HYPERPARAMETER_TUNING:
        selected = max(
            candidate_results,
            key=lambda result: result["selection_score"],
        )
        _print_tuning_results(
            "External-model hyperparameter tuning on all B-cell data",
            candidate_results,
            index_key="candidate_id",
            parameters_key="parameters",
            metric_key="selection_metric",
            score_key="selection_score",
            selected_result=selected,
        )
    else:
        selected = candidate_results[0]
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
        "candidate_count": len(candidates),
    }
    return selection, comparison, selected["oof_scores"]
