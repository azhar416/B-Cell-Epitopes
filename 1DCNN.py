from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from CFG import CFG
from dataset import prepare_dataset_frames, stratified_kfold_indices
from train import _binary_metrics, _build_loss_function
import oversampling


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
UNKNOWN_TOKEN_ID = len(AMINO_ACIDS) + 1
RESIDUE_TO_ID = {
    residue: index + 1
    for index, residue in enumerate(AMINO_ACIDS)
}
RESIDUE_TO_ID.update({residue: UNKNOWN_TOKEN_ID for residue in "XBUOZ"})
PADDING_TOKEN_ID = 0


class CNNPeptideDataset(Dataset):
    """Fixed-length residue IDs for training the CNN."""

    def __init__(self, dataframe, max_length):
        if max_length < 1:
            raise ValueError("max_length must be a positive integer.")
        if dataframe[["peptide_seq", "target"]].isna().any().any():
            raise ValueError("Sequences and labels must not contain missing values.")

        self.input_ids = torch.full(
            (len(dataframe), max_length),
            PADDING_TOKEN_ID,
            dtype=torch.long,
        )
        labels = pd.to_numeric(dataframe["target"], errors="raise")
        if not labels.isin((0, 1)).all():
            raise ValueError("'target' must contain only binary labels 0 and 1.")
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.float32)

        allowed_residues = set(RESIDUE_TO_ID)
        for row_index, value in enumerate(dataframe["peptide_seq"]):
            sequence = "".join(str(value).upper().split())
            if not sequence:
                raise ValueError(f"Empty peptide sequence at row {row_index}.")
            invalid = set(sequence).difference(allowed_residues)
            if invalid:
                raise ValueError(
                    f"Unsupported residues at row {row_index}: {sorted(invalid)}"
                )
            residue_ids = [
                RESIDUE_TO_ID[residue]
                for residue in sequence[:max_length]
            ]
            self.input_ids[row_index, : len(residue_ids)] = torch.tensor(
                residue_ids,
                dtype=torch.long,
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(PADDING_TOKEN_ID),
            "labels": self.labels[index],
        }


class CNN1DForClassification(nn.Module):
    """Parallel 1D convolutions over amino-acid residue embeddings."""

    def __init__(
        self,
        embedding_dim=32,
        channels=64,
        kernel_sizes=(3, 5, 7),
        dropout=0.3,
    ):
        super().__init__()
        if not kernel_sizes or any(size < 1 or size % 2 == 0 for size in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers.")
        self.embedding = nn.Embedding(
            num_embeddings=UNKNOWN_TOKEN_ID + 1,
            embedding_dim=embedding_dim,
            padding_idx=PADDING_TOKEN_ID,
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=channels,
                kernel_size=size,
                padding=size // 2,
            )
            for size in kernel_sizes
        )
        pooled_features = channels * len(kernel_sizes) * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_features, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(PADDING_TOKEN_ID)
        mask = attention_mask.bool().unsqueeze(1)
        embedded = self.embedding(input_ids).transpose(1, 2)

        pooled = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            masked_features = features.masked_fill(
                ~mask,
                torch.finfo(features.dtype).min,
            )
            maximum = masked_features.max(dim=2).values
            mean = (features * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
            pooled.extend((maximum, mean))
        return self.classifier(torch.cat(pooled, dim=1)).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_cnn_datasets(cfg=None, data_dir=None):
    """Prepare CNN inputs for row-level stratified cross-validation."""
    cfg = cfg or CFG()
    if data_dir is None:
        configured_path = Path(cfg._DATASET_PATH)
        data_dir = configured_path if configured_path.exists() else "dataset"
    dataframe = prepare_dataset_frames(
        dataset_name=cfg.DATASET_NAME,
        data_dir=data_dir,
    )
    return CNNPeptideDataset(dataframe, max_length=cfg.MAX_LENGTH)


def _evaluate(model, loader, criterion, cfg, device):
    model.eval()
    total_loss = 0.0
    example_count = 0
    targets = []
    probabilities = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            predictions = model(**inputs)
            loss = criterion(predictions, labels)
            total_loss += loss.item() * labels.size(0)
            example_count += labels.size(0)
            targets.append(labels.cpu().numpy())
            probabilities.append(predictions.cpu().numpy())

    if not example_count:
        raise ValueError("Cannot evaluate an empty dataset.")
    metrics = _binary_metrics(
        targets,
        probabilities,
        cfg.PREDICTION_THRESHOLD,
    )
    return {"loss": total_loss / example_count, **metrics}


def train_cnn(
    train_dataset,
    cfg=None,
    model=None,
    device=None,
    checkpoint_path="checkpoints/1dcnn_classifier.pt",
    seed=None,
    validation_dataset=None,
    fold_number=None,
    n_splits=None,
):
    """Train for fixed epochs and optionally log validation metrics each epoch."""
    cfg = cfg or CFG()
    if cfg.EPOCHS < 1 or cfg.BATCH_SIZE < 1:
        raise ValueError("EPOCHS and BATCH_SIZE must be at least 1.")
    if not len(train_dataset):
        raise ValueError("Training dataset must be non-empty.")

    set_seed(cfg.SEED if seed is None else seed)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model or CNN1DForClassification()
    model.to(device)

    if oversampling.USE_OVERSAMPLING:
        original_size = len(train_dataset)
        train_dataset = oversampling.oversample_training_dataset(
            train_dataset,
            random_state=cfg.SEED if seed is None else seed,
        )
        print(
            "Training-only random oversampling enabled: "
            f"{original_size} -> {len(train_dataset)} examples"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=device.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )
    criterion = _build_loss_function(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.HEAD_LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    output_path = Path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0
        example_count = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            optimizer.zero_grad(set_to_none=True)
            predictions = model(**inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.GRADIENT_CLIP_NORM,
            )
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            example_count += labels.size(0)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / example_count,
        }
        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = _evaluate(
                model,
                validation_loader,
                criterion,
                cfg,
                device,
            )
            epoch_record.update(
                {
                    f"validation_{name}": validation_metrics[name]
                    for name in (
                        "loss",
                        "auc",
                        "recall",
                        "precision",
                        "accuracy",
                    )
                }
            )
        history.append(epoch_record)
        fold_label = (
            f"Fold {fold_number}/{n_splits} | "
            if fold_number is not None
            else ""
        )
        print(
            f"{fold_label}Epoch {epoch + 1}/{cfg.EPOCHS} - "
            f"train_loss: {epoch_record['train_loss']:.4f}"
            + (
                ""
                if validation_metrics is None
                else (
                    f", validation_loss: {validation_metrics['loss']:.4f}, "
                    f"validation_auc: {validation_metrics['auc']:.4f}, "
                    f"validation_recall: {validation_metrics['recall']:.4f}, "
                    f"validation_precision: {validation_metrics['precision']:.4f}, "
                    f"validation_accuracy: {validation_metrics['accuracy']:.4f}"
                )
            )
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "epochs_trained": len(history),
            "model_type": "1dcnn",
            "max_length": cfg.MAX_LENGTH,
        },
        output_path,
    )
    print(f"Saved checkpoint to {output_path}")
    return model, history


def cross_validate_cnn(
    dataset,
    cfg=None,
    device=None,
    n_splits=None,
    model_factory=CNN1DForClassification,
    checkpoint_path="checkpoints/1dcnn_classifier.pt",
):
    """Train and evaluate a fresh CNN in each stratified row-level fold."""
    cfg = cfg or CFG()
    n_splits = cfg.N_SPLITS if n_splits is None else n_splits
    folds = stratified_kfold_indices(
        dataset.labels.detach().cpu().numpy(),
        n_splits=n_splits,
        random_state=cfg.SEED,
    )
    base_checkpoint = Path(checkpoint_path)
    fold_metrics = []
    evaluation_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    for fold_number, (train_indices, evaluation_indices) in enumerate(folds, 1):
        print(
            f"\nFold {fold_number}/{n_splits}: "
            f"train={len(train_indices):,} ({len(train_indices) / len(dataset):.0%}), "
            f"evaluation={len(evaluation_indices):,} "
            f"({len(evaluation_indices) / len(dataset):.0%})"
        )
        fold_checkpoint = base_checkpoint.with_name(
            f"{base_checkpoint.stem}_fold_{fold_number}{base_checkpoint.suffix}"
        )
        model, _ = train_cnn(
            train_dataset=Subset(dataset, train_indices.tolist()),
            cfg=cfg,
            model=model_factory(),
            device=device,
            checkpoint_path=fold_checkpoint,
            seed=cfg.SEED + fold_number - 1,
            validation_dataset=Subset(dataset, evaluation_indices.tolist()),
            fold_number=fold_number,
            n_splits=n_splits,
        )
        evaluation_loader = DataLoader(
            Subset(dataset, evaluation_indices.tolist()),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=evaluation_device.type == "cuda",
        )
        metrics = _evaluate(
            model.to(evaluation_device),
            evaluation_loader,
            _build_loss_function(cfg),
            cfg,
            evaluation_device,
        )
        precision = metrics["precision"]
        recall = metrics["recall"]
        metrics["f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
        fold_metrics.append({"fold": fold_number, **metrics})

    metric_names = ("loss", "auc", "accuracy", "precision", "recall", "f1")
    mean_metrics = {
        name: float(np.mean([fold[name] for fold in fold_metrics]))
        for name in metric_names
    }
    std_metrics = {
        name: float(np.std([fold[name] for fold in fold_metrics], ddof=1))
        for name in metric_names
    }
    print("\nStratified k-fold results:")
    for name in metric_names:
        print(f"  {name}: {mean_metrics[name]:.4f} +/- {std_metrics[name]:.4f}")
    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }


def evaluate_cnn(model, test_dataset, cfg=None, device=None):
    """Evaluate a trained CNN on a dataset."""
    cfg = cfg or CFG()
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    metrics = _evaluate(
        model.to(device),
        test_loader,
        _build_loss_function(cfg),
        cfg,
        device,
    )
    precision = metrics["precision"]
    recall = metrics["recall"]
    return {
        "test_loss": metrics["loss"],
        "auc": metrics["auc"],
        "accuracy": metrics["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def main():
    cfg = CFG()
    dataset = prepare_cnn_datasets(cfg)
    cross_validate_cnn(dataset, cfg=cfg)


if __name__ == "__main__":
    main()