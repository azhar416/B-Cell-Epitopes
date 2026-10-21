from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from CFG import CFG
from model import ProtBertForClassification


class BinaryFocalLoss(nn.Module):
    """Binary focal loss for probabilities produced by a sigmoid model."""

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probabilities, targets):
        probabilities = probabilities.reshape(-1).clamp(
            min=torch.finfo(probabilities.dtype).eps,
            max=1 - torch.finfo(probabilities.dtype).eps,
        )
        targets = targets.to(dtype=probabilities.dtype).reshape(-1)
        if probabilities.shape != targets.shape:
            raise ValueError("Probabilities and targets must have matching shapes.")

        binary_cross_entropy = nn.functional.binary_cross_entropy(
            probabilities,
            targets,
            reduction="none",
        )
        probability_of_correct_class = torch.exp(-binary_cross_entropy)
        alpha_factor = torch.where(
            targets == 1,
            self.alpha,
            1 - self.alpha,
        )
        loss = alpha_factor * (1 - probability_of_correct_class).pow(
            self.gamma
        ) * binary_cross_entropy
        return loss.mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(dataset, cfg, shuffle, device):
    return DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )


def _binary_metrics(targets, probabilities, threshold):
    targets = np.concatenate(targets).astype(np.int64, copy=False)
    probabilities = np.concatenate(probabilities)
    predictions = probabilities >= threshold
    positives = targets == 1
    negatives = ~positives

    true_positive = int(np.count_nonzero(predictions & positives))
    true_negative = int(np.count_nonzero(~predictions & negatives))
    false_positive = int(np.count_nonzero(predictions & negatives))
    false_negative = int(np.count_nonzero(~predictions & positives))

    positive_count = int(positives.sum())
    negative_count = len(targets) - positive_count
    if positive_count and negative_count:
        order = np.argsort(probabilities, kind="mergesort")
        sorted_probabilities = probabilities[order]
        _, starts, counts = np.unique(
            sorted_probabilities,
            return_index=True,
            return_counts=True,
        )
        ranks = np.empty(len(targets), dtype=np.float64)
        for start, count in zip(starts, counts):
            ranks[order[start : start + count]] = (
                start + 1 + (count - 1) / 2
            )
        auc = (
            ranks[positives].sum() - positive_count * (positive_count + 1) / 2
        ) / (positive_count * negative_count)
    else:
        auc = float("nan")

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(positive_count, 1)
    accuracy = (true_positive + true_negative) / max(len(targets), 1)
    return {
        "auc": float(auc),
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
    }


def train_model(train_dataset, cfg=None, model=None, device=None):
    """Train ProtBert on a prepared dataset and save the final checkpoint."""
    cfg = cfg or CFG()
    if cfg.EPOCHS < 1:
        raise ValueError("EPOCHS must be at least 1.")
    if cfg.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")

    set_seed(cfg.SEED)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model or ProtBertForClassification(
        model_name=cfg.MODEL_NAME,
        dropout=cfg.DROPOUT,
    )
    if isinstance(model, nn.DataParallel):
        model = model.module

    if cfg.FREEZE_ENCODER:
        for parameter in model.bert.parameters():
            parameter.requires_grad = False
        model.bert.eval()

    device_ids = []
    if (
        cfg.USE_MULTI_GPU
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        primary_device = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        device_ids = [primary_device] + [
            index
            for index in range(torch.cuda.device_count())
            if index != primary_device
        ]
        device = torch.device("cuda", primary_device)

    model.to(device)
    train_loader = _make_loader(train_dataset, cfg, shuffle=True, device=device)
    criterion = BinaryFocalLoss(
        alpha=cfg.FOCAL_ALPHA,
        gamma=cfg.FOCAL_GAMMA,
    )

    if cfg.FREEZE_ENCODER:
        optimizer_parameters = [
            {
                "params": model.classifier.parameters(),
                "lr": cfg.HEAD_LEARNING_RATE,
            }
        ]
    else:
        optimizer_parameters = [
            {
                "params": model.bert.parameters(),
                "lr": cfg.ENCODER_LEARNING_RATE,
            },
            {
                "params": model.classifier.parameters(),
                "lr": cfg.HEAD_LEARNING_RATE,
            },
        ]

    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    if device_ids:
        model = nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0],
        )
        print(f"Training across {len(device_ids)} GPUs: {device_ids}")
    else:
        print(f"Training on {device}")

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    history = []
    model.train()
    for epoch in range(cfg.EPOCHS):
        if cfg.FREEZE_ENCODER:
            base_model.bert.eval()
        total_loss = 0.0
        example_count = 0
        epoch_targets = []
        epoch_probabilities = []
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.EPOCHS}",
            unit="batch",
        )
        for batch in progress_bar:
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)
            probabilities = model(**inputs)
            loss = criterion(probabilities, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.GRADIENT_CLIP_NORM,
            )
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            example_count += batch_size
            epoch_targets.append(labels.detach().cpu().numpy())
            epoch_probabilities.append(probabilities.detach().cpu().numpy())
            metrics = _binary_metrics(
                epoch_targets,
                epoch_probabilities,
                cfg.PREDICTION_THRESHOLD,
            )
            progress_bar.set_postfix(
                batch_loss=f"{loss.item():.4f}",
                avg_loss=f"{total_loss / example_count:.4f}",
                auc=f"{metrics['auc']:.4f}",
                recall=f"{metrics['recall']:.4f}",
                precision=f"{metrics['precision']:.4f}",
                accuracy=f"{metrics['accuracy']:.4f}",
            )

        epoch_loss = total_loss / example_count
        metrics = _binary_metrics(
            epoch_targets,
            epoch_probabilities,
            cfg.PREDICTION_THRESHOLD,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_auc": metrics["auc"],
                "train_recall": metrics["recall"],
                "train_precision": metrics["precision"],
                "train_accuracy": metrics["accuracy"],
            }
        )
        print(
            f"Epoch {epoch + 1}/{cfg.EPOCHS} - "
            f"loss: {epoch_loss:.4f}, "
            f"AUC: {metrics['auc']:.4f}, "
            f"recall: {metrics['recall']:.4f}, "
            f"precision: {metrics['precision']:.4f}, "
            f"accuracy: {metrics['accuracy']:.4f}"
        )

    checkpoint_path = Path(cfg.CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "model_name": cfg.MODEL_NAME,
            "dropout": cfg.DROPOUT,
            "freeze_encoder": cfg.FREEZE_ENCODER,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    return model, history


def evaluate_model(model, test_dataset, cfg=None, device=None):
    """Evaluate a trained model once on the held-out test dataset."""
    cfg = cfg or CFG()
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    model.eval()
    test_loader = _make_loader(test_dataset, cfg, shuffle=False, device=device)
    criterion = BinaryFocalLoss(
        alpha=cfg.FOCAL_ALPHA,
        gamma=cfg.FOCAL_GAMMA,
    )

    total_loss = 0.0
    example_count = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    with torch.inference_mode():
        for batch in test_loader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            probabilities = model(**inputs)
            loss = criterion(probabilities, labels)
            predictions = probabilities >= cfg.PREDICTION_THRESHOLD
            actual = labels >= cfg.PREDICTION_THRESHOLD

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            example_count += batch_size
            true_positive += int((predictions & actual).sum().item())
            true_negative += int((~predictions & ~actual).sum().item())
            false_positive += int((predictions & ~actual).sum().item())
            false_negative += int((~predictions & actual).sum().item())

    if example_count == 0:
        raise ValueError("Cannot evaluate an empty test dataset.")

    accuracy = (true_positive + true_negative) / example_count
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "test_loss": total_loss / example_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }