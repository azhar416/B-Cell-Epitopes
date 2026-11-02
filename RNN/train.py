import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from RNN.CFG import CFG
from RNN.dataset import (
    TABULAR_FEATURE_COLUMNS,
    PeptideSequenceDataset,
    collate_sequences,
    load_dataset,
    scale_tabular_features,
)
from RNN.model import BiLSTMClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stratified_split(labels, validation_size=0.2, random_state=42):
    """Match the label-stratified 80:20 holdout used by the other workflows."""
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("A stratified split requires binary labels.")
    train_indices, validation_indices = train_test_split(
        np.arange(labels.size),
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )
    return train_indices, validation_indices


def calculate_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
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
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _evaluate(model, data_loader, criterion, device, threshold):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key != "labels"
            }
            logits = model(**inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(torch.sigmoid(logits).cpu().numpy())
    if not all_labels:
        raise ValueError("Cannot evaluate an empty dataset.")
    metrics = calculate_metrics(all_labels, all_probabilities, threshold)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics, np.asarray(all_labels), np.asarray(all_probabilities)


def run_training(config=None):
    config = config or CFG()
    config.validate()
    set_seed(config.RANDOM_STATE)
    device = torch.device(
        "cuda"
        if config.DEVICE == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.DEVICE == "auto"
        else config.DEVICE
    )
    dataframe = load_dataset(
        dataset_name=config.DATASET_NAME,
        data_dir=config.DATA_DIR,
        max_peptide_length=config.MAX_PEPTIDE_LENGTH,
        flanking_size=config.FLANKING_SIZE,
    )
    dataset = PeptideSequenceDataset(dataframe)
    labels = dataset.labels.numpy().astype(np.int64)
    train_indices, validation_indices = make_stratified_split(
        labels,
        validation_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
    )
    imputer, scaler = scale_tabular_features(dataset, train_indices)

    train_subset = Subset(dataset, train_indices.tolist())
    validation_subset = Subset(dataset, validation_indices.tolist())
    loader_settings = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "collate_fn": collate_sequences,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_subset, shuffle=True, **loader_settings)
    train_eval_loader = DataLoader(train_subset, shuffle=False, **loader_settings)
    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        **loader_settings,
    )

    model = BiLSTMClassifier(
        tabular_feature_count=len(TABULAR_FEATURE_COLUMNS),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss()
    history = []
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    print(
        f"RNN split: train={len(train_indices):,} "
        f"({len(train_indices) / len(dataset):.0%}), "
        f"validation={len(validation_indices):,} "
        f"({len(validation_indices) / len(dataset):.0%}); device={device}."
    )
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            labels_batch = batch["labels"].to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key != "labels"
            }
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs)
            loss = criterion(logits, labels_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels_batch.size(0)
            seen += labels_batch.size(0)

        train_metrics, _, _ = _evaluate(
            model,
            train_eval_loader,
            criterion,
            device,
            config.PREDICTION_THRESHOLD,
        )
        validation_metrics, _, _ = _evaluate(
            model,
            validation_loader,
            criterion,
            device,
            config.PREDICTION_THRESHOLD,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "optimization_loss": running_loss / seen,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        val_auc = validation_metrics["roc_auc"]
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"train loss={train_metrics['loss']:.4f}, "
            f"train ROC-AUC={train_metrics['roc_auc']:.4f}, "
            f"validation loss={validation_metrics['loss']:.4f}, "
            f"validation ROC-AUC={val_auc:.4f}, "
            f"validation PR-AUC={validation_metrics['pr_auc']:.4f}"
        )
        if val_auc > best_auc + config.EARLY_STOPPING_MIN_DELTA:
            best_auc = val_auc
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping: validation ROC-AUC stopped improving.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")
    model.load_state_dict(best_state)
    validation_metrics, validation_labels, validation_probabilities = _evaluate(
        model,
        validation_loader,
        criterion,
        device,
        config.PREDICTION_THRESHOLD,
    )

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "bilstm_classifier.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(config).items()
            },
            "tabular_feature_columns": list(TABULAR_FEATURE_COLUMNS),
            "imputer_statistics": imputer.statistics_,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "best_epoch": best_epoch,
            "best_validation_roc_auc": best_auc,
            "history": history,
        },
        checkpoint_path,
    )
    metrics_path = output_dir / "validation_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "validation_metrics": validation_metrics,
                "validation_labels": validation_labels.astype(int).tolist(),
                "validation_probabilities": validation_probabilities.tolist(),
                "best_epoch": best_epoch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Best epoch: {best_epoch}; checkpoint: {checkpoint_path}")
    print(f"Validation metrics saved to {metrics_path}")
    return {
        "model": model,
        "history": history,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "validation_metrics": validation_metrics,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
    }