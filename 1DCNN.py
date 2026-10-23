from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from CFG import CFG
from dataset import prepare_dataset_frames
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
    """Prepare CNN inputs from the exact grouped split used by dataset.py."""
    cfg = cfg or CFG()
    if data_dir is None:
        configured_path = Path(cfg._DATASET_PATH)
        data_dir = configured_path if configured_path.exists() else "dataset"
    frames = prepare_dataset_frames(
        dataset_name=cfg.DATASET_NAME,
        data_dir=data_dir,
        train_size=cfg.TRAIN_SIZE,
        validation_size=cfg.VALIDATION_SIZE,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.SEED,
    )
    return tuple(
        CNNPeptideDataset(frame, max_length=cfg.MAX_LENGTH)
        for frame in frames
    )


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
    validation_dataset,
    cfg=None,
    model=None,
    device=None,
    checkpoint_path="checkpoints/1dcnn_classifier.pt",
):
    """Train with the project's validation-AUC early-stopping settings."""
    cfg = cfg or CFG()
    if cfg.EPOCHS < 1 or cfg.BATCH_SIZE < 1:
        raise ValueError("EPOCHS and BATCH_SIZE must be at least 1.")
    if not len(train_dataset) or not len(validation_dataset):
        raise ValueError("Training and validation datasets must be non-empty.")
    if cfg.EARLY_STOPPING_PATIENCE < 1:
        raise ValueError("EARLY_STOPPING_PATIENCE must be at least 1.")
    if cfg.EARLY_STOPPING_MIN_DELTA < 0:
        raise ValueError("EARLY_STOPPING_MIN_DELTA must be non-negative.")

    set_seed(cfg.SEED)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model or CNN1DForClassification()
    model.to(device)

    if oversampling.USE_OVERSAMPLING:
        original_size = len(train_dataset)
        train_dataset = oversampling.oversample_training_dataset(
            train_dataset,
            random_state=cfg.SEED,
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
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
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
    best_auc = float("-inf")
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

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

        validation_metrics = _evaluate(
            model,
            validation_loader,
            criterion,
            cfg,
            device,
        )
        validation_auc = validation_metrics["auc"]
        if not np.isfinite(validation_auc):
            raise ValueError(
                "Validation data must contain both target classes to monitor AUC."
            )
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / example_count,
            "validation_loss": validation_metrics["loss"],
            "validation_auc": validation_auc,
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch + 1}/{cfg.EPOCHS} - "
            f"train_loss: {epoch_record['train_loss']:.4f}, "
            f"val_loss: {validation_metrics['loss']:.4f}, "
            f"val_auc: {validation_auc:.4f}"
        )

        improved = validation_auc > best_auc + cfg.EARLY_STOPPING_MIN_DELTA
        tied_with_better_loss = (
            best_epoch > 0
            and validation_auc == best_auc
            and validation_metrics["loss"] < best_validation_loss
        )
        if improved or tied_with_better_loss:
            best_auc = validation_auc
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch + 1
            epochs_without_improvement = (
                0 if improved else epochs_without_improvement + 1
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_validation_auc": best_auc,
                    "best_validation_loss": best_validation_loss,
                    "model_type": "1dcnn",
                    "max_length": cfg.MAX_LENGTH,
                },
                output_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping; best validation AUC {best_auc:.4f} "
                f"at epoch {best_epoch}."
            )
            break

    checkpoint = torch.load(output_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Restored best model from epoch {best_epoch} "
        f"(validation AUC {best_auc:.4f}); checkpoint: {output_path}"
    )
    return model, history


def evaluate_cnn(model, test_dataset, cfg=None, device=None):
    """Evaluate the trained CNN once on the held-out test split."""
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
    train_dataset, validation_dataset, test_dataset = prepare_cnn_datasets(cfg)
    model, _ = train_cnn(
        train_dataset,
        validation_dataset,
        cfg=cfg,
    )
    test_metrics = evaluate_cnn(model, test_dataset, cfg=cfg)
    print("Held-out test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()