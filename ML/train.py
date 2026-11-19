"""Command-line entry point and orchestration for classical ML training."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from ML.CFG import CFG
from ML.evaluation import (
    calculate_metrics,
    save_evaluation_plots,
    save_train_validation_metrics_plot,
    select_threshold_by_mcc,
)
from ML.dataset import (
    make_cross_validation_splits,
    load_labeled_dataset,
    make_grouped_split,
    make_stratified_split,
)
from ML.oversampling import smote_resample
from ML.selection import (
    _build_configured_model,
    _select_hyperparameters_on_validation,
    _select_external_model_from_bcell,
    apply_parameters,
)
from ML.pre_dataset import build_feature_frame


def _parent_protein_id_mapping(
    dataframe: pd.DataFrame, config: CFG
) -> dict[str, int] | None:
    """Stable protein-id codes, which only perbandingan_3 feeds to the model.

    Codes come from the sorted unique ids of the frame they are derived from,
    so the same development set always yields the same mapping.
    """
    if config.FEATURE_SET != "perbandingan_3":
        return None
    identifiers = sorted(dataframe["parent_protein_id"].astype(str).unique())
    return {identifier: index for index, identifier in enumerate(identifiers)}


#: One console line per fitted model: display name, metric key, format spec.
_REPORTED_METRICS = (
    ("ROC-AUC", "roc_auc", ".4f"),
    ("PR-AUC", "pr_auc", ".4f"),
    ("Accuracy", "accuracy", ".4f"),
    ("Recall", "recall", ".4f"),
    ("Precision", "precision", ".4f"),
    ("F1", "f1", ".4f"),
    ("TN", "tn", "d"),
    ("FP", "fp", "d"),
    ("FN", "fn", "d"),
    ("TP", "tp", "d"),
    ("Balanced accuracy", "balanced_accuracy", ".4f"),
    ("MCC", "mcc", ".4f"),
)

#: Rows of the external-evaluation comparison table.
_SUMMARY_METRICS = (
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC (AP)"),
    ("accuracy", "Accuracy"),
    ("recall", "Recall / Sens."),
    ("specificity", "Specificity"),
    ("precision", "Precision"),
    ("f1", "F1-Score"),
    ("mcc", "MCC"),
)


def _format_metrics(label: str, metrics: dict[str, float | int]) -> str:
    """One-line console summary of a fitted model's metrics."""
    fields = ", ".join(
        f"{name}={metrics[key]:{spec}}" for name, key, spec in _REPORTED_METRICS
    )
    return f"{label} {fields}"


def _mean_and_std(
    rows: list[dict[str, object]], skip: tuple[str, ...] = ()
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-metric mean and population SD across equally shaped metric dicts."""
    names = [key for key in rows[0] if key not in skip]
    return (
        {name: float(np.mean([row[name] for row in rows])) for name in names},
        {name: float(np.std([row[name] for row in rows])) for name in names},
    )


def _load_dataset(config: CFG, dataset_name: str, data_dir: Path) -> pd.DataFrame:
    return load_labeled_dataset(
        dataset_name,
        data_dir,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
        feature_set=config.FEATURE_SET,
        bert_embedding_dir=config.BERT_EMBEDDING_DIR,
        prottrans_embedding_dir=config.PROTTRANS_EMBEDDING_DIR,
    )


def _build_configured_features(
    dataframe: pd.DataFrame,
    config: CFG,
    parent_protein_id_mapping: dict[str, int] | None = None,
) -> pd.DataFrame:
    if parent_protein_id_mapping is None:
        parent_protein_id_mapping = _parent_protein_id_mapping(dataframe, config)
    return build_feature_frame(
        dataframe,
        use_peptide_features=config.USE_PEPTIDE_FEATURES,
        use_protein_features=config.USE_PROTEIN_FEATURES,
        use_biological_features=config.USE_BIOLOGICAL_FEATURES,
        peptide_feature_type=config.PEPTIDE_FEATURE_TYPE,
        feature_set=config.FEATURE_SET,
        use_aac_features=config.EFFECTIVE_AAC_FEATURES,
        use_dpc_features=config.EFFECTIVE_DPC_FEATURES,
        parent_protein_id_mapping=parent_protein_id_mapping,
        use_flank_features=config.USE_FLANK_FEATURES,
        flank_window=config.FLANK_WINDOW,
        flank_split_sides=config.FLANK_SPLIT_SIDES,
        use_len_features=config.USE_LEN_FEATURES,
        use_prop_features=config.USE_PROP_FEATURES,
        use_protglobal_features=config.USE_PROTGLOBAL_FEATURES,
        use_protlen_features=config.USE_PROTLEN_FEATURES,
        use_pos_features=config.USE_POS_FEATURES,
        use_protid_features=config.USE_PROTID_FEATURES,
    )


def _audit_exact_overlap(
    bcell: pd.DataFrame, sars: pd.DataFrame
) -> tuple[dict[str, int], set[str]]:
    """Confirm no cleaned B-cell peptide reappears in SARS, and record it.

    Cleaning removes the overlap from the development set, so this is a guard
    rather than a disclosure: it must come back empty. It is kept, and kept
    loud, because the whole external claim rests on that emptiness, and a
    future change to the loader would otherwise break it in silence.
    """
    overlap_mask = bcell["peptide_seq"].isin(set(sars["peptide_seq"]))
    overlap_sequences = set(bcell.loc[overlap_mask, "peptide_seq"])
    bcell_labels = dict(zip(bcell["peptide_seq"], bcell["target"]))
    sars_labels = dict(zip(sars["peptide_seq"], sars["target"]))
    same_label = sum(
        bcell_labels[sequence] == sars_labels[sequence]
        for sequence in overlap_sequences
    )
    audit = {
        "unique_exact_peptide_sequences": len(overlap_sequences),
        "same_label": int(same_label),
        "opposite_label": int(len(overlap_sequences) - same_label),
        "bcell_rows_retained_in_development": int(overlap_mask.sum()),
        "sars_rows_retained_for_external_evaluation": int(len(sars)),
    }
    if overlap_sequences:
        raise RuntimeError(
            f"{len(overlap_sequences):,} peptide(s) appear in both the "
            "development set and SARS, so the external evaluation would not "
            "be on unseen peptides. Cleaning is supposed to have removed "
            "them; ML.dataset.load_labeled_dataset is where that happens."
        )
    return audit, overlap_sequences


def _align_external_features(
    config: CFG, development: pd.DataFrame, sars: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int] | None]:
    """Build both feature matrices under one schema and one protein mapping."""
    mapping = _parent_protein_id_mapping(development, config)
    development_features = _build_configured_features(development, config, mapping)
    sars_features = _build_configured_features(sars, config, mapping)
    missing_features = development_features.columns.difference(sars_features.columns)
    extra_features = sars_features.columns.difference(development_features.columns)
    if len(missing_features) or len(extra_features):
        raise ValueError(
            "B-cell and SARS feature schemas differ after feature generation. "
            f"Missing in SARS: {missing_features.tolist()}; "
            f"extra in SARS: {extra_features.tolist()}."
        )
    return (
        development_features,
        sars_features.reindex(columns=development_features.columns),
        mapping,
    )


def _internal_cv_for_external(
    config: CFG,
    development: pd.DataFrame,
    output_dir: Path,
    selection: tuple[dict[str, object], list[dict[str, object]]],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run the same_dataset CV runner on B-cell and realign its OOF scores.

    The hyperparameter selection made for the external model is handed over
    rather than recomputed: both arms select on the same rows, folds and
    candidates, so running it twice would only burn another 45 fits.
    """
    internal_config = copy.copy(config)
    internal_config.TRAINING_MODE = "same_dataset"
    internal_config.DATASET_NAME = "bcell"
    internal_config.OUTPUT_DIR = output_dir / "internal_bcell_same_dataset"
    report = run_training(internal_config, hyperparameter_selection=selection)

    key_columns = ["peptide_seq", "target", "parent_protein_id"]
    out_of_fold = pd.read_csv(report["out_of_fold_predictions"])
    out_of_fold = development[key_columns].merge(
        out_of_fold[key_columns + ["validation_probability", "fold"]],
        on=key_columns,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if out_of_fold["validation_probability"].isna().any():
        raise RuntimeError(
            "The same_dataset internal CV did not produce one OOF score for "
            "every cleaned B-cell peptide."
        )
    return report, out_of_fold


def _print_external_summary(
    *,
    selected_threshold: float,
    internal_mean: dict[str, float],
    internal_std: dict[str, float],
    internal_pooled: dict[str, float | int],
    sars_at_05: dict[str, float | int],
    sars_at_selected: dict[str, float | int],
) -> None:
    """Print the internal-vs-external comparison table transcribed into the paper."""
    print("\n" + "=" * 96)
    print("EXPERIMENT EVALUATION SUMMARY: B-cell 5-Fold Group CV vs External SARS")
    print("=" * 96)
    print(
        f"{'Metric':<16} | {'B-cell CV (Mean ± SD)':<24} | "
        f"{'B-cell Pooled OOF':<18} | {'SARS (Thresh 0.5)':<18} | "
        f"{'SARS (Thresh ' + f'{selected_threshold:.3f})':<20}"
    )
    print("-" * 96)
    for key, name in _SUMMARY_METRICS:
        print(
            f"{name:<16} | {internal_mean.get(key, 0.0):.4f} ± "
            f"{internal_std.get(key, 0.0):.4f}          | "
            f"{internal_pooled.get(key, 0.0):<18.4f} | "
            f"{sars_at_05.get(key, 0.0):<18.4f} | "
            f"{sars_at_selected.get(key, 0.0):<20.4f}"
        )
    print("=" * 96 + "\n")


def _run_external_sars(config: CFG) -> dict[str, object]:
    """Fit/choose exclusively on B-cell, then evaluate all SARS rows once."""
    bcell = _load_dataset(config, "bcell", config.DATA_DIR)
    sars = _load_dataset(
        config, "sars", config.EVALUATION_DATA_DIR or config.DATA_DIR
    )
    overlap_report, overlap_sequences = _audit_exact_overlap(bcell, sars)

    # Retain the complete cleaned B-cell dataset, including exact peptide
    # overlaps with SARS, for both internal selection and the final refit.
    development = bcell.reset_index(drop=True)
    labels = development["target"].to_numpy(dtype=int)
    groups = development["parent_protein_id"].astype(str).to_numpy()
    if np.unique(labels).size != 2:
        raise ValueError(
            "The cleaned B-cell development set must contain both classes."
        )

    development_features, sars_features, parent_protein_id_mapping = (
        _align_external_features(config, development, sars)
    )

    selection, candidate_comparison, selected_model_bcell_oof_scores = (
        _select_external_model_from_bcell(
            config, development_features, labels, groups
        )
    )
    final_config = apply_parameters(config, selection["parameters"])

    fit_features, fit_labels, synthetic_count = smote_resample(
        development_features,
        labels,
        enabled=config.USE_SMOTE,
        random_state=config.RANDOM_STATE,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
    )
    final_model = _build_configured_model(final_config, fit_labels)
    final_model.fit(fit_features, fit_labels)
    sars_scores = final_model.predict_proba(sars_features)[:, 1]
    sars_labels = sars["target"].to_numpy(dtype=int)
    output_dir = config.OUTPUT_DIR / config.FEATURE_SET / "external_sars"
    output_dir.mkdir(parents=True, exist_ok=True)

    internal_cv_report, internal_oof = _internal_cv_for_external(
        config, development, output_dir, (selection, candidate_comparison)
    )
    bcell_oof_scores = internal_oof["validation_probability"].to_numpy(dtype=float)
    internal_oof_labels = internal_oof["target"].to_numpy(dtype=int)
    internal_oof_metrics = internal_cv_report["validation_metrics"]
    internal_oof_metrics_at_05 = calculate_metrics(
        internal_oof_labels, bcell_oof_scores, threshold=0.5
    )

    # The folds and the selection sweep refit the same candidate on the same
    # rows, so their out-of-fold scores must agree; a drift here would mean the
    # two arms have silently diverged.
    oof_disagreement = float(
        np.abs(bcell_oof_scores - selected_model_bcell_oof_scores).max()
    )
    if oof_disagreement > 1e-6:
        print(
            "WARNING: internal-CV and selection out-of-fold scores differ by "
            f"up to {oof_disagreement:.3e}; the two arms are not running the "
            "same configuration."
        )

    # Select the deployment threshold using OOF probabilities from the same
    # globally selected hyperparameter configuration used for the final model.
    selected_threshold, threshold_selection_mcc = select_threshold_by_mcc(
        labels, selected_model_bcell_oof_scores
    )
    internal_oof_metrics_selected = calculate_metrics(
        internal_oof_labels, bcell_oof_scores, threshold=selected_threshold
    )
    internal_fold_metrics_selected = [
        {
            "fold": int(fold_number),
            **calculate_metrics(
                fold_frame["target"].to_numpy(dtype=int),
                fold_frame["validation_probability"].to_numpy(dtype=float),
                threshold=selected_threshold,
            ),
        }
        for fold_number, fold_frame in internal_oof.groupby("fold", sort=True)
    ]
    internal_fold_metrics = [
        {
            "fold": fold["fold"],
            **fold["validation_metrics"],
            "selected_hyperparameters": fold["selected_hyperparameters"],
        }
        for fold in internal_cv_report["fold_metrics"]
    ]
    internal_mean_metrics_selected, internal_std_metrics_selected = _mean_and_std(
        internal_fold_metrics_selected, skip=("fold",)
    )

    pd.DataFrame(
        {
            "peptide_seq": internal_oof["peptide_seq"],
            "target": internal_oof["target"],
            "parent_protein_id": internal_oof["parent_protein_id"],
            "bcell_oof_probability": bcell_oof_scores,
            "fold": internal_oof["fold"],
        }
    ).to_csv(output_dir / "bcell_oof_predictions.csv", index=False)
    pd.DataFrame(internal_fold_metrics).assign(
        selected_hyperparameters=lambda frame: frame[
            "selected_hyperparameters"
        ].map(json.dumps)
    ).to_csv(output_dir / "bcell_fold_metrics.csv", index=False)
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
    report: dict[str, object] = {
        "training_mode": "external_sars",
        "training_dataset": "bcell",
        "external_evaluation_dataset": "sars",
        "external_evaluation_split": "all cleaned SARS rows evaluated once; no SARS split",
        "sars_labels_used_for_model_or_threshold_selection": False,
        "internal_selection_protocol": (
            "5-fold StratifiedGroupKFold on B-cell; each outer validation fold "
            "directly scores the hyperparameter candidates, so these internal "
            "metrics are tuning results rather than an independent estimate."
        ),
        "primary_selection_metric": (
            "average_precision"
            if config.HYPERPARAMETER_TUNING_METRIC == "pr_auc"
            else config.HYPERPARAMETER_TUNING_METRIC
        )
        if config.HYPERPARAMETER_TUNING
        else None,
        "model": config.MODEL_TYPE,
        "feature_set": config.FEATURE_SET,
        "feature_columns": development_features.columns.tolist(),
        "feature_selection_method": (
            "paper_reported_rfe_subset"
            if config.FEATURE_SET == "perbandingan_3"
            else None
        ),
        "bcell_rows_before_overlap_handling": int(len(bcell)),
        "bcell_rows_used_for_development": int(len(development)),
        "bcell_proteins_used_for_development": int(development.parent_protein_id.nunique()),
        "bcell_class_counts_used_for_development": pd.Series(labels).value_counts().sort_index().to_dict(),
        "external_sars_rows": int(len(sars)),
        "external_sars_proteins": int(sars.parent_protein_id.nunique()),
        "external_sars_class_counts": pd.Series(sars_labels).value_counts().sort_index().to_dict(),
        "exact_sequence_overlap_audit": overlap_report,
        "exact_sequence_overlap_handling": (
            "Cleaning removes every B-cell peptide that also appears in SARS, "
            "for internal selection and final training alike, so no peptide "
            "scored in the external evaluation was ever trained on. SARS "
            "itself is kept whole."
        ),
        "internal_bcell_cv": {
            "protocol": internal_cv_report["validation_metrics_interpretation"],
            "folds": internal_fold_metrics,
            "mean_metrics": internal_cv_report["mean_validation_metrics"],
            "std_metrics": internal_cv_report["std_validation_metrics"],
            "folds_at_selected_threshold": internal_fold_metrics_selected,
            "mean_metrics_at_selected_threshold": internal_mean_metrics_selected,
            "std_metrics_at_selected_threshold": internal_std_metrics_selected,
            "oof_metrics_threshold_configured": internal_oof_metrics,
            "oof_metrics_threshold_0_5": internal_oof_metrics_at_05,
            "oof_metrics_bcell_selected_threshold": internal_oof_metrics_selected,
            "configured_threshold": config.PREDICTION_THRESHOLD,
            "selected_threshold": selected_threshold,
            "threshold_selection_source": (
                "Pooled B-cell out-of-fold predictions from the globally "
                "selected hyperparameter candidate."
            ),
            "threshold_selection_mcc": threshold_selection_mcc,
            "hyperparameter_comparison_path": internal_cv_report[
                "hyperparameter_comparison_path"
            ],
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
        "model_selection": {
            "parameters": selection["parameters"],
            "selection_metric": selection["selection_metric"],
            "selection_score": selection["selection_score"],
            "candidate_count": selection["candidate_count"],
            "candidate_comparison_path": str(
                output_dir / "bcell_model_selection.csv"
            ),
        },
        "resampling": {
            "method": "SMOTE" if config.USE_SMOTE else "none",
            "scope": "B-cell training folds and final B-cell refit only",
            "synthetic_rows_final_refit": int(synthetic_count),
        },
        "class_weight": config.CLASS_WEIGHT,
        "prediction_threshold": selected_threshold,
        "threshold_selection_source": (
            "Pooled B-cell out-of-fold predictions from the globally selected "
            "hyperparameter candidate."
        ),
        "external_sars_metrics_threshold_0_5": metrics_at_05,
        "external_sars_metrics_bcell_selected_threshold": metrics_at_selected,
        "evaluation_plots": sars_plots,
        "predictions_path": str(output_dir / "external_sars_predictions.csv"),
        "artifact_directory": str(output_dir),
        "interpretation_limit": (
            "The cleaned SARS dataset contains one parent protein; this is a "
            "single-protein external evaluation, not broad unseen-protein "
            "validation. Peptides shared with B-cell were removed from the "
            "development set, so every SARS row scored here is unseen."
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
        "parent_protein_id_mapping": parent_protein_id_mapping,
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_aac_features": config.EFFECTIVE_AAC_FEATURES,
        "use_dpc_features": config.EFFECTIVE_DPC_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "use_flank_features": config.USE_FLANK_FEATURES,
        "flank_window": config.FLANK_WINDOW,
        "flank_split_sides": config.FLANK_SPLIT_SIDES,
        "use_len_features": config.USE_LEN_FEATURES,
        "use_prop_features": config.USE_PROP_FEATURES,
        "use_protglobal_features": config.USE_PROTGLOBAL_FEATURES,
        "use_protlen_features": config.USE_PROTLEN_FEATURES,
        "use_pos_features": config.USE_POS_FEATURES,
        "use_protid_features": config.USE_PROTID_FEATURES,
        "peptide_feature_type": config.PEPTIDE_FEATURE_TYPE,
        "prediction_threshold": selected_threshold,
        "training_dataset": "bcell",
        "external_evaluation_dataset": "sars",
        "selected_hyperparameters": selection["parameters"],
        "exact_overlap_sequences_retained_in_training": len(overlap_sequences),
        "development_excludes_sars_peptides": True,
    }
    dump(bundle, output_dir / "peptide_classifier.joblib")

    _print_external_summary(
        selected_threshold=selected_threshold,
        internal_mean=internal_cv_report["mean_validation_metrics"],
        internal_std=internal_cv_report["std_validation_metrics"],
        internal_pooled=internal_oof_metrics,
        sars_at_05=metrics_at_05,
        sars_at_selected=metrics_at_selected,
    )
    print(
        f"External SARS evaluation complete: {len(sars):,} peptides from "
        f"{sars.parent_protein_id.nunique():,} protein(s), none of whose "
        "peptides appear in the development set."
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

    dataframe = _load_dataset(config, config.DATASET_NAME, config.DATA_DIR)
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
    selected_hyperparameters, hyperparameter_comparison, selected_model = (
        _select_hyperparameters_on_validation(
            config=config,
            fit_features=fit_features,
            fit_labels=fit_labels,
            validation_features=feature_frame.iloc[validation_indices],
            validation_labels=labels[validation_indices],
        )
    )
    fitted_config = apply_parameters(config, selected_hyperparameters)

    # Tuning already fitted the winner on exactly these rows and parameters.
    if selected_model is not None:
        model = selected_model
    else:
        model = _build_configured_model(fitted_config, fit_labels)
        model.fit(fit_features, fit_labels)
    train_probabilities = model.predict_proba(train_features)[:, 1]
    validation_probabilities = model.predict_proba(
        feature_frame.iloc[validation_indices]
    )[:, 1]
    train_metrics = calculate_metrics(
        train_labels,
        train_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
    )
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
    print(_format_metrics("Training", train_metrics))
    print(_format_metrics("Validation", validation_metrics))

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
        "parent_protein_id_mapping": _parent_protein_id_mapping(dataframe, config),
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_aac_features": config.EFFECTIVE_AAC_FEATURES,
        "use_dpc_features": config.EFFECTIVE_DPC_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "use_flank_features": config.USE_FLANK_FEATURES,
        "flank_window": config.FLANK_WINDOW,
        "flank_split_sides": config.FLANK_SPLIT_SIDES,
        "use_len_features": config.USE_LEN_FEATURES,
        "use_prop_features": config.USE_PROP_FEATURES,
        "use_protglobal_features": config.USE_PROTGLOBAL_FEATURES,
        "use_protlen_features": config.USE_PROTLEN_FEATURES,
        "use_pos_features": config.USE_POS_FEATURES,
        "use_protid_features": config.USE_PROTID_FEATURES,
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
        "feature_selection_method": (
            "paper_reported_rfe_subset"
            if config.FEATURE_SET == "perbandingan_3"
            else None
        ),
        "class_labels": {"0": "negative", "1": "positive"},
        "use_peptide_features": config.USE_PEPTIDE_FEATURES,
        "use_aac_features": config.EFFECTIVE_AAC_FEATURES,
        "use_dpc_features": config.EFFECTIVE_DPC_FEATURES,
        "use_protein_features": config.USE_PROTEIN_FEATURES,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "use_flank_features": config.USE_FLANK_FEATURES,
        "flank_window": config.FLANK_WINDOW,
        "flank_split_sides": config.FLANK_SPLIT_SIDES,
        "use_len_features": config.USE_LEN_FEATURES,
        "use_prop_features": config.USE_PROP_FEATURES,
        "use_protglobal_features": config.USE_PROTGLOBAL_FEATURES,
        "use_protlen_features": config.USE_PROTLEN_FEATURES,
        "use_pos_features": config.USE_POS_FEATURES,
        "use_protid_features": config.USE_PROTID_FEATURES,
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


def run_training(
    config: CFG,
    hyperparameter_selection: tuple[
        dict[str, object], list[dict[str, object]]
    ] | None = None,
) -> dict[str, object]:
    """Run five-fold CV, grouping proteins except for peptide-level SARS folds.

    ``hyperparameter_selection`` lets a caller that has already run the global
    selection on exactly this dataset hand the result over instead of paying
    for it twice; the external arm is the only such caller.
    """
    config.validate()
    config.OUTPUT_DIR = Path(config.OUTPUT_DIR)
    if config.TRAINING_MODE == "external_sars":
        return _run_external_sars(config)

    dataframe = _load_dataset(config, config.DATASET_NAME, config.DATA_DIR)
    folds, split_strategy = make_cross_validation_splits(
        dataframe["target"].to_numpy(dtype=int),
        dataframe["parent_protein_id"].to_numpy(),
        dataset_name=config.DATASET_NAME,
        n_splits=5,
        random_state=config.RANDOM_STATE,
    )
    profile_output_dir = config.OUTPUT_DIR / config.FEATURE_SET
    summary_dir = profile_output_dir / "cross_validation"

    # One hyperparameter set is chosen on pooled out-of-fold predictions over
    # the whole development set, exactly as the external arm does, and then
    # held fixed across all five folds. Both phases therefore select under the
    # same protocol, and the reported mean +/- SD describes a single model
    # instead of averaging five differently-tuned ones.
    global_selection: dict[str, object] | None = None
    global_comparison: list[dict[str, object]] = []
    fold_base_config = config
    if config.HYPERPARAMETER_TUNING:
        if hyperparameter_selection is not None:
            global_selection, global_comparison = hyperparameter_selection
            source = "Reusing hyperparameters selected by the caller"
        else:
            global_selection, global_comparison, _ = (
                _select_external_model_from_bcell(
                    config,
                    _build_configured_features(dataframe, config),
                    dataframe["target"].to_numpy(dtype=int),
                    dataframe["parent_protein_id"].astype(str).to_numpy(),
                    dataset_name=config.DATASET_NAME,
                )
            )
            source = "Globally selected hyperparameters"
        fold_base_config = apply_parameters(config, global_selection["parameters"])
        print(
            f"{source} "
            f"({global_selection['selection_metric']}="
            f"{global_selection['selection_score']:.6f}): "
            f"{global_selection['parameters']}"
        )

    fold_reports = []
    for fold_number, fold_split in enumerate(folds, 1):
        fold_config = copy.copy(fold_base_config)
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

    # Candidates are now compared once, globally, so the comparison table has
    # one row per candidate rather than one per (fold, candidate) pair.
    hyperparameter_comparison_rows = list(global_comparison)

    out_of_fold_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = summary_dir / "out_of_fold_predictions.csv"
    out_of_fold_predictions.to_csv(predictions_path, index=False)
    hyperparameter_comparison_path = None
    if config.HYPERPARAMETER_TUNING and hyperparameter_comparison_rows:
        hyperparameter_comparison_path = (
            summary_dir / "hyperparameter_comparison.csv"
        )
        comparison_frame = pd.DataFrame(hyperparameter_comparison_rows)
        comparison_frame["parameters"] = comparison_frame["parameters"].map(
            json.dumps
        )
        comparison_frame.to_csv(hyperparameter_comparison_path, index=False)
    pooled_metrics = calculate_metrics(
        out_of_fold_predictions["target"].to_numpy(dtype=int),
        out_of_fold_predictions["validation_probability"].to_numpy(dtype=float),
        threshold=config.PREDICTION_THRESHOLD,
    )
    mean_validation_metrics, std_validation_metrics = _mean_and_std(
        [report["validation_metrics"] for report in fold_reports]
    )
    mean_training_metrics, std_training_metrics = _mean_and_std(
        [report["training_metrics"] for report in fold_reports]
    )
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
        "hyperparameter_selection_protocol": (
            "One hyperparameter set selected on pooled out-of-fold predictions "
            "over the whole development set, then held fixed across all five "
            "folds. This matches the external arm's protocol."
            if config.HYPERPARAMETER_TUNING
            else "No tuning; configured hyperparameters used as-is."
        ),
        "selected_hyperparameters": (
            global_selection["parameters"] if global_selection else None
        ),
        "hyperparameter_tuning_warning": (
            "Hyperparameters were selected on the same out-of-fold predictions "
            "these metrics are computed from, so internal CV numbers remain "
            "model-selection results rather than an independent performance "
            "estimate. Use a separate untouched dataset for final evaluation."
            if config.HYPERPARAMETER_TUNING
            else None
        ),
        "out_of_fold_predictions": str(predictions_path),
        "artifact_directory": str(summary_dir),
        "fold_reports": fold_reports,
        "validation_metrics_interpretation": (
            "Pooled predictions from five folds, all using one globally "
            "selected hyperparameter set. Selection used these same "
            "out-of-fold predictions, so the metrics are model-selection "
            "results, not an independent performance estimate. SARS folds "
            "stratify peptides and share the single source protein."
            if config.DATASET_NAME.lower() == "sars"
            else "Pooled predictions from five stratified protein-group folds, "
            "all using one globally selected hyperparameter set. Selection "
            "used these same out-of-fold predictions, so the metrics are "
            "model-selection results, not an independent performance estimate."
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
        choices=("same_dataset", "external_sars"),
        help=(
            "Use internal dataset CV or train/select on B-cell only and "
            "evaluate all cleaned SARS rows once."
        ),
    )
    parser.add_argument(
        "--model",
        choices=(
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "svm",
            "xgboost",
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=(
            "perbandingan_1",
            "perbandingan_1b",
            "perbandingan_2",
            "perbandingan_3",
            "perbandingan_4",
            "perbandingan_5",
            "proposed",
        ),
        help=(
            "Select a paper comparison feature profile or the current proposed "
            "feature engineering."
        ),
    )
    parser.add_argument(
        "--bert-embedding-dir",
        type=Path,
        help=(
            "Directory containing precomputed <dataset>_bert_cls.csv files "
            "for perbandingan_4."
        ),
    )
    parser.add_argument(
        "--prottrans-embedding-dir",
        type=Path,
        help=(
            "Directory containing bcell_prott5.npz and sars_prott5.npz "
            "precomputed numerical embeddings for perbandingan_5."
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
    if args.bert_embedding_dir:
        config.BERT_EMBEDDING_DIR = args.bert_embedding_dir
    if args.prottrans_embedding_dir:
        config.PROTTRANS_EMBEDDING_DIR = args.prottrans_embedding_dir
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
