"""Train ProtT5-embedding SVM or Random Forest models."""

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from EmbeddingML.CFG import CFG
from EmbeddingML.dataset import (
    assemble_features,
    make_cross_validation_splits,
    make_dataset_split,
)
from EmbeddingML.encoder import load_or_create_embeddings
from EmbeddingML.evaluate import calculate_metrics, save_evaluation_plots
from EmbeddingML.model import build_model
from EmbeddingML.preprocessing import load_labeled_dataset


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _prediction_frame(dataframe, labels, probabilities, threshold):
    return pd.DataFrame(
        {
            "source_row_index": dataframe["source_row_index"].to_numpy(),
            "parent_protein_id": dataframe["parent_protein_id"].to_numpy(),
            "peptide_seq": dataframe["peptide_seq"].to_numpy(),
            "target": labels,
            "probability": probabilities,
            "prediction": (probabilities >= threshold).astype(np.int64),
        }
    )


def _run_training_fold(
    config: CFG | None = None,
    *,
    encoder=None,
    fold_split: tuple[np.ndarray, np.ndarray] | None = None,
    split_strategy: str | None = None,
    fold_number: int | None = None,
) -> dict[str, object]:
    config = config or CFG()
    config.validate()
    dataframe = load_labeled_dataset(
        dataset_name=config.DATASET_NAME,
        data_dir=config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
    )
    if fold_split is None:
        train_indices, validation_indices, split_strategy = make_dataset_split(
            dataframe,
            dataset_name=config.DATASET_NAME,
            validation_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_STATE,
        )
    else:
        train_indices, validation_indices = fold_split
        split_strategy = split_strategy or "cross_validation"
    labels = dataframe["target"].to_numpy(dtype=np.int64)
    sequences = dataframe["peptide_seq"].tolist()

    dataset_output_dir = Path(config.OUTPUT_DIR) / config.DATASET_NAME
    model_output_dir = dataset_output_dir / config.MODEL_TYPE
    if fold_number is not None:
        model_output_dir /= "cross_validation" / f"fold_{fold_number}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dataset_output_dir / "prott5_embeddings.npz"
    embeddings = load_or_create_embeddings(
        sequences,
        model_name=config.ENCODER_NAME,
        tokenizer_name=config.TOKENIZER_NAME,
        cache_path=cache_path,
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
        force_recompute=config.FORCE_RECOMPUTE_EMBEDDINGS,
        encoder=encoder,
    )
    features, feature_names = assemble_features(
        embeddings,
        dataframe,
        use_biological_features=config.USE_BIOLOGICAL_FEATURES,
    )

    model = build_model(config)
    model.fit(features[train_indices], labels[train_indices])
    train_probabilities = model.predict_proba(features[train_indices])[:, 1]
    validation_probabilities = model.predict_proba(
        features[validation_indices]
    )[:, 1]
    train_metrics = calculate_metrics(
        labels[train_indices],
        train_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
    )
    validation_metrics = calculate_metrics(
        labels[validation_indices],
        validation_probabilities,
        threshold=config.PREDICTION_THRESHOLD,
    )
    plots = save_evaluation_plots(
        labels[validation_indices],
        validation_probabilities,
        output_dir=model_output_dir,
        threshold=config.PREDICTION_THRESHOLD,
    )

    split_labels = np.full(len(dataframe), "unused", dtype=object)
    split_labels[train_indices] = "train"
    split_labels[validation_indices] = "validation"
    split_frame = dataframe[
        ["source_row_index", "parent_protein_id", "peptide_seq", "target"]
    ].copy()
    split_frame["partition"] = split_labels
    split_frame.to_csv(model_output_dir / "split_assignments.csv", index=False)
    _prediction_frame(
        dataframe.iloc[train_indices],
        labels[train_indices],
        train_probabilities,
        config.PREDICTION_THRESHOLD,
    ).to_csv(model_output_dir / "training_predictions.csv", index=False)
    _prediction_frame(
        dataframe.iloc[validation_indices],
        labels[validation_indices],
        validation_probabilities,
        config.PREDICTION_THRESHOLD,
    ).to_csv(model_output_dir / "validation_predictions.csv", index=False)
    dataframe.to_csv(model_output_dir / "cleaned_dataset.csv", index=False)

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "encoder_name": config.ENCODER_NAME,
        "tokenizer_name": config.TOKENIZER_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "prediction_threshold": config.PREDICTION_THRESHOLD,
        "dataset_name": config.DATASET_NAME,
        "split_strategy": split_strategy,
        "max_peptide_length": config.MAX_PEPTIDE_LENGTH,
    }
    model_path = model_output_dir / "model.joblib"
    dump(bundle, model_path)

    group_ids = dataframe["parent_protein_id"].to_numpy()
    train_groups = set(group_ids[train_indices])
    validation_groups = set(group_ids[validation_indices])
    group_overlap = bool(train_groups.intersection(validation_groups))
    report: dict[str, object] = {
        "dataset_name": config.DATASET_NAME,
        "fold": fold_number,
        "model_type": config.MODEL_TYPE,
        "encoder_name": config.ENCODER_NAME,
        "tokenizer_name": config.TOKENIZER_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "pooling": "mean over non-special ProtT5 residue tokens",
        "feature_count": int(features.shape[1]),
        "feature_names": feature_names,
        "use_biological_features": config.USE_BIOLOGICAL_FEATURES,
        "split_strategy": split_strategy,
        "requested_validation_fraction": config.VALIDATION_SIZE,
        "train_size": int(len(train_indices)),
        "validation_size": int(len(validation_indices)),
        "train_fraction": float(len(train_indices) / len(dataframe)),
        "validation_fraction": float(len(validation_indices) / len(dataframe)),
        "train_protein_count": len(train_groups),
        "validation_protein_count": len(validation_groups),
        "train_validation_proteins_overlap": group_overlap,
        "train_class_counts": (
            pd.Series(labels[train_indices]).value_counts().sort_index().to_dict()
        ),
        "validation_class_counts": (
            pd.Series(labels[validation_indices])
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "training_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "prediction_threshold": config.PREDICTION_THRESHOLD,
        "validation_metrics_interpretation": (
            "Protein-group holdout: no parent protein occurs in both partitions."
            if not group_overlap
            else (
                "Peptide-level stratified holdout: train and validation share "
                "the same source protein; this is not evaluation on unseen proteins."
            )
        ),
        "model_path": str(model_path),
        "embedding_cache_path": str(cache_path),
        "plots": plots,
        "validation_predictions": str(
            model_output_dir / "validation_predictions.csv"
        ),
        "configuration": _json_safe(asdict(config)),
    }
    metrics_path = model_output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Split: {split_strategy}; train={len(train_indices):,} "
        f"({len(train_indices) / len(dataframe):.1%}), "
        f"validation={len(validation_indices):,} "
        f"({len(validation_indices) / len(dataframe):.1%}); "
        f"proteins={len(train_groups):,}/{len(validation_groups):,}; "
        f"overlap={group_overlap}."
    )
    print(
        f"Validation ROC-AUC={validation_metrics['roc_auc']:.4f}, "
        f"PR-AUC={validation_metrics['pr_auc']:.4f}, "
        f"F1={validation_metrics['f1']:.4f}"
    )
    print(f"Saved model and report to: {model_output_dir}")
    return report


def run_training(
    config: CFG | None = None,
    *,
    encoder=None,
) -> dict[str, object]:
    """Train and evaluate five stratified folds; use peptide folds for SARS."""
    config = config or CFG()
    config.validate()
    dataframe = load_labeled_dataset(
        dataset_name=config.DATASET_NAME,
        data_dir=config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
    )
    folds, split_strategy = make_cross_validation_splits(
        dataframe,
        dataset_name=config.DATASET_NAME,
        n_splits=5,
        random_state=config.RANDOM_STATE,
    )
    fold_reports = []
    for fold_number, fold_split in enumerate(folds, 1):
        fold_config = copy.copy(config)
        fold_reports.append(
            _run_training_fold(
                fold_config,
                encoder=encoder,
                fold_split=fold_split,
                split_strategy=split_strategy,
                fold_number=fold_number,
            )
        )

    prediction_frames = []
    for fold_number, fold_report in enumerate(fold_reports, 1):
        fold_predictions = pd.read_csv(fold_report["validation_predictions"])
        fold_predictions["fold"] = fold_number
        prediction_frames.append(fold_predictions)
    out_of_fold_predictions = pd.concat(prediction_frames, ignore_index=True)

    model_output_dir = (
        Path(config.OUTPUT_DIR)
        / config.DATASET_NAME
        / config.MODEL_TYPE
        / "cross_validation"
    )
    model_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_output_dir / "out_of_fold_predictions.csv"
    out_of_fold_predictions.to_csv(predictions_path, index=False)
    pooled_metrics = calculate_metrics(
        out_of_fold_predictions["target"].to_numpy(dtype=np.int64),
        out_of_fold_predictions["probability"].to_numpy(dtype=np.float64),
        threshold=config.PREDICTION_THRESHOLD,
    )
    pooled_plots = save_evaluation_plots(
        out_of_fold_predictions["target"].to_numpy(dtype=np.int64),
        out_of_fold_predictions["probability"].to_numpy(dtype=np.float64),
        output_dir=model_output_dir,
        threshold=config.PREDICTION_THRESHOLD,
    )
    fold_metric_names = fold_reports[0]["validation_metrics"]
    mean_validation_metrics = {
        metric: float(
            np.mean(
                [fold["validation_metrics"][metric] for fold in fold_reports]
            )
        )
        for metric in fold_metric_names
    }
    std_validation_metrics = {
        metric: float(
            np.std(
                [fold["validation_metrics"][metric] for fold in fold_reports]
            )
        )
        for metric in fold_metric_names
    }
    training_metric_names = fold_reports[0]["training_metrics"]
    mean_training_metrics = {
        metric: float(
            np.mean([fold["training_metrics"][metric] for fold in fold_reports])
        )
        for metric in training_metric_names
    }
    std_training_metrics = {
        metric: float(
            np.std([fold["training_metrics"][metric] for fold in fold_reports])
        )
        for metric in training_metric_names
    }
    report: dict[str, object] = {
        "dataset_name": config.DATASET_NAME,
        "model_type": config.MODEL_TYPE,
        "split_strategy": split_strategy,
        "n_splits": 5,
        "train_size": int(
            np.mean([fold["train_size"] for fold in fold_reports])
        ),
        "validation_size": int(len(out_of_fold_predictions)),
        "train_fraction": float(
            np.mean([fold["train_fraction"] for fold in fold_reports])
        ),
        "validation_fraction": float(
            np.mean([fold["validation_fraction"] for fold in fold_reports])
        ),
        "train_protein_count": float(
            np.mean([fold["train_protein_count"] for fold in fold_reports])
        ),
        "validation_protein_count": float(
            np.mean([fold["validation_protein_count"] for fold in fold_reports])
        ),
        "train_validation_proteins_overlap": (
            config.DATASET_NAME.lower() == "sars"
        ),
        "train_class_counts": (
            dataframe["target"].value_counts().sort_index().to_dict()
        ),
        "validation_class_counts": (
            out_of_fold_predictions["target"].value_counts().sort_index().to_dict()
        ),
        "training_metrics": mean_training_metrics,
        "mean_training_metrics": mean_training_metrics,
        "std_training_metrics": std_training_metrics,
        "validation_metrics": pooled_metrics,
        "mean_validation_metrics": mean_validation_metrics,
        "std_validation_metrics": std_validation_metrics,
        "fold_metrics": [
            {
                "fold": index,
                "training_metrics": fold["training_metrics"],
                "validation_metrics": fold["validation_metrics"],
                "train_size": fold["train_size"],
                "validation_size": fold["validation_size"],
            }
            for index, fold in enumerate(fold_reports, 1)
        ],
        "validation_metrics_interpretation": (
            "Pooled out-of-fold predictions from five stratified peptide folds. "
            "The same SARS protein occurs in all folds; this is not evaluation "
            "on unseen proteins."
            if config.DATASET_NAME.lower() == "sars"
            else "Pooled out-of-fold predictions from five stratified "
            "protein-group folds with no protein crossing folds."
        ),
        "out_of_fold_predictions": str(predictions_path),
        "plots": pooled_plots,
        "artifact_directory": str(model_output_dir),
        "fold_artifact_directories": [
            fold["model_path"] for fold in fold_reports
        ],
    }
    metrics_path = model_output_dir / "cross_validation_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"5-fold CV complete ({split_strategy}); "
        f"pooled validation ROC-AUC={pooled_metrics['roc_auc']:.4f}, "
        f"PR-AUC={pooled_metrics['pr_auc']:.4f}."
    )
    print(f"Saved cross-validation metrics to: {metrics_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SVM or Random Forest on frozen ProtT5 embeddings."
    )
    parser.add_argument("--dataset", choices=("bcell", "sars"))
    parser.add_argument("--model", choices=("svm", "random_forest"))
    parser.add_argument("--use-biological-features", action="store_true")
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "cuda:0"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CFG()
    if args.dataset:
        config.DATASET_NAME = args.dataset
    if args.model:
        config.MODEL_TYPE = args.model
    if args.use_biological_features:
        config.USE_BIOLOGICAL_FEATURES = True
    if args.force_recompute_embeddings:
        config.FORCE_RECOMPUTE_EMBEDDINGS = True
    if args.device:
        config.DEVICE = args.device
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    run_training(config)


if __name__ == "__main__":
    main()