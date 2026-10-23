from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from CFG import CFG
from model import ESM2ForClassification, ProtBertForClassification
import oversampling


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


def _build_loss_function(cfg):
    loss_name = cfg.LOSS_FUNCTION.lower()
    if loss_name == "bce":
        return nn.BCELoss()
    if loss_name == "binary_focal":
        return BinaryFocalLoss(
            alpha=cfg.FOCAL_ALPHA,
            gamma=cfg.FOCAL_GAMMA,
        )
    raise ValueError(
        "LOSS_FUNCTION must be either 'bce' or 'binary_focal'; "
        f"received {cfg.LOSS_FUNCTION!r}."
    )


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


def train_model(
    train_dataset,
    cfg=None,
    model=None,
    device=None,
    validation_dataset=None,
):
    """Train a protein language model with validation-AUC early stopping."""
    cfg = cfg or CFG()
    if cfg.EPOCHS < 1:
        raise ValueError("EPOCHS must be at least 1.")
    if cfg.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")
    if validation_dataset is None or len(validation_dataset) == 0:
        raise ValueError("A non-empty validation_dataset is required.")
    if cfg.EARLY_STOPPING_PATIENCE < 1:
        raise ValueError("EARLY_STOPPING_PATIENCE must be at least 1.")
    if cfg.EARLY_STOPPING_MIN_DELTA < 0:
        raise ValueError("EARLY_STOPPING_MIN_DELTA must be non-negative.")

    set_seed(cfg.SEED)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_type = cfg.MODEL_TYPE.lower()
    model_classes = {
        "protbert": ProtBertForClassification,
        "esm2": ESM2ForClassification,
    }
    if model_type not in model_classes:
        raise ValueError("MODEL_TYPE must be either 'protbert' or 'esm2'.")
    if len(train_dataset) == 0:
        raise ValueError("A non-empty train_dataset is required.")
    print("Training configuration:")
    print(f"  Model: {cfg.MODEL_TYPE} ({cfg.MODEL_NAME})")
    print(f"  Dataset: {cfg.DATASET_NAME}")
    print(
        "  Split sizes: "
        f"train={cfg.TRAIN_SIZE:.0%}, "
        f"validation={cfg.VALIDATION_SIZE:.0%}, "
        f"test={cfg.TEST_SIZE:.0%} "
        f"(examples: train={len(train_dataset):,}, "
        f"validation={len(validation_dataset):,})"
    )
    print(f"  Loss function: {cfg.LOSS_FUNCTION}")
    if cfg.LOSS_FUNCTION.lower() == "binary_focal":
        print(
            f"    Focal alpha={cfg.FOCAL_ALPHA}, gamma={cfg.FOCAL_GAMMA}"
        )
    print(f"  LoRA: {'enabled' if cfg.USE_LORA else 'disabled'}")
    print(
        "  LoRA config: "
        f"rank={cfg.LORA_RANK}, alpha={cfg.LORA_ALPHA}, "
        f"dropout={cfg.LORA_DROPOUT}"
        if cfg.USE_LORA
        else "  LoRA config: not active"
    )
    print(
        "  Learning rates: "
        f"head={cfg.HEAD_LEARNING_RATE:g}, "
        f"LoRA={cfg.LORA_LEARNING_RATE:g}"
        if cfg.USE_LORA
        else f"  Learning rate: head={cfg.HEAD_LEARNING_RATE:g}"
    )
    print(f"  Epochs: {cfg.EPOCHS}")

    model = model or model_classes[model_type](
        model_name=cfg.MODEL_NAME,
        use_lora=cfg.USE_LORA,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        lora_target_modules=cfg.LORA_TARGET_MODULES,
    )
    if isinstance(model, nn.DataParallel):
        model = model.module
    print("Model architecture:")
    print(model)
    model_uses_lora = getattr(model, "use_lora", False)
    if cfg.USE_LORA and not model_uses_lora:
        raise ValueError(
            "USE_LORA is enabled, but the supplied model was not created with LoRA."
        )
    if not cfg.USE_LORA and model_uses_lora:
        raise ValueError(
            "The supplied model uses LoRA, but USE_LORA is disabled in CFG."
        )

    if cfg.USE_LORA:
        adapter_parameters = []
        for name, parameter in model.bert.named_parameters():
            is_adapter = "lora_" in name
            parameter.requires_grad = is_adapter
            if is_adapter:
                adapter_parameters.append(parameter)
        if not adapter_parameters:
            raise ValueError("LoRA is enabled, but no adapter parameters were found.")
    elif cfg.FREEZE_ENCODER:
        for parameter in model.bert.parameters():
            parameter.requires_grad = False
        model.bert.eval()
    else:
        for parameter in model.bert.parameters():
            parameter.requires_grad = True

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
    else:
        print("Training-only random oversampling disabled.")
    train_loader = _make_loader(train_dataset, cfg, shuffle=True, device=device)
    validation_loader = _make_loader(
        validation_dataset,
        cfg,
        shuffle=False,
        device=device,
    )
    criterion = _build_loss_function(cfg)

    encoder_parameters = [
        parameter
        for parameter in model.bert.parameters()
        if parameter.requires_grad
    ]
    classifier_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]
    optimizer_parameters = []
    if encoder_parameters:
        optimizer_parameters.append(
            {
                "params": encoder_parameters,
                "lr": (
                    cfg.LORA_LEARNING_RATE
                    if cfg.USE_LORA
                    else cfg.ENCODER_LEARNING_RATE
                ),
            }
        )
    if classifier_parameters:
        optimizer_parameters.append(
            {
                "params": classifier_parameters,
                "lr": cfg.HEAD_LEARNING_RATE,
            }
        )
    if not optimizer_parameters:
        raise ValueError("The model has no trainable parameters.")

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
    best_auc = float("-inf")
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_auc_improvement = 0
    checkpoint_path = Path(cfg.CHECKPOINT_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.EPOCHS):
        model.train()
        if cfg.FREEZE_ENCODER and not cfg.USE_LORA:
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
        validation_metrics = _evaluate_dataset(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            cfg=cfg,
            device=device,
        )
        validation_auc = validation_metrics["auc"]
        validation_loss = validation_metrics["loss"]
        if not np.isfinite(validation_auc):
            raise ValueError(
                "Validation data must contain both target classes to monitor AUC."
            )
        auc_improved = validation_auc > best_auc + cfg.EARLY_STOPPING_MIN_DELTA
        auc_tied_with_better_loss = (
            best_epoch > 0
            and validation_auc == best_auc
            and validation_loss < best_validation_loss
        )
        if auc_improved:
            best_auc = validation_auc
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_auc_improvement = 0
        elif auc_tied_with_better_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_auc_improvement += 1
        else:
            epochs_without_auc_improvement += 1

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_auc": metrics["auc"],
                "train_recall": metrics["recall"],
                "train_precision": metrics["precision"],
                "train_accuracy": metrics["accuracy"],
                "validation_loss": validation_loss,
                "validation_auc": validation_auc,
                "validation_recall": validation_metrics["recall"],
                "validation_precision": validation_metrics["precision"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
        )
        print(
            f"Epoch {epoch + 1}/{cfg.EPOCHS} - "
            f"train_loss: {epoch_loss:.4f}, "
            f"train_auc: {metrics['auc']:.4f}, "
            f"val_loss: {validation_loss:.4f}, "
            f"val_auc: {validation_auc:.4f}, "
            f"val_recall: {validation_metrics['recall']:.4f}, "
            f"val_precision: {validation_metrics['precision']:.4f}, "
            f"val_accuracy: {validation_metrics['accuracy']:.4f}"
        )
        if auc_improved or auc_tied_with_better_loss:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=base_model,
                optimizer=optimizer,
                history=history,
                cfg=cfg,
                best_epoch=best_epoch,
                best_validation_auc=best_auc,
                best_validation_loss=best_validation_loss,
            )
        if epochs_without_auc_improvement >= cfg.EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping after "
                f"{epoch + 1} epochs; best validation AUC "
                f"{best_auc:.4f} at epoch {best_epoch}."
            )
            break

    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    base_model.load_state_dict(best_checkpoint["model_state_dict"])
    best_checkpoint["history"] = history
    torch.save(best_checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Restored best model from epoch {best_epoch} (validation AUC {best_auc:.4f}).")
    return model, history


def _evaluate_dataset(model, data_loader, criterion, cfg, device):
    model.eval()
    total_loss = 0.0
    example_count = 0
    all_targets = []
    all_probabilities = []
    with torch.inference_mode():
        for batch in data_loader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            probabilities = model(**inputs)
            loss = criterion(probabilities, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            example_count += batch_size
            all_targets.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    if example_count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    metrics = _binary_metrics(
        all_targets,
        all_probabilities,
        cfg.PREDICTION_THRESHOLD,
    )
    return {
        "loss": total_loss / example_count,
        **metrics,
    }


def _save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    history,
    cfg,
    best_epoch,
    best_validation_auc,
    best_validation_loss,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "model_name": cfg.MODEL_NAME,
            "model_type": cfg.MODEL_TYPE,
            "loss_function": cfg.LOSS_FUNCTION,
            "freeze_encoder": cfg.FREEZE_ENCODER,
            "use_lora": cfg.USE_LORA,
            "lora_rank": cfg.LORA_RANK if cfg.USE_LORA else None,
            "lora_alpha": cfg.LORA_ALPHA if cfg.USE_LORA else None,
            "lora_dropout": cfg.LORA_DROPOUT if cfg.USE_LORA else None,
            "lora_target_modules": (
                cfg.LORA_TARGET_MODULES if cfg.USE_LORA else None
            ),
            "best_epoch": best_epoch,
            "best_validation_auc": best_validation_auc,
            "best_validation_loss": best_validation_loss,
        },
        checkpoint_path,
    )


def evaluate_model(model, test_dataset, cfg=None, device=None):
    """Evaluate a trained model once on the held-out test dataset."""
    cfg = cfg or CFG()
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    test_loader = _make_loader(test_dataset, cfg, shuffle=False, device=device)
    test_metrics = _evaluate_dataset(
        model=model,
        data_loader=test_loader,
        criterion=_build_loss_function(cfg),
        cfg=cfg,
        device=device,
    )
    precision = test_metrics["precision"]
    recall = test_metrics["recall"]
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "test_loss": test_metrics["loss"],
        "auc": test_metrics["auc"],
        "accuracy": test_metrics["accuracy"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_training_results(history, test_metrics, save_path=None):
    """Plot epoch training metrics alongside final held-out test metrics."""
    if not history:
        raise ValueError("history must contain at least one training epoch.")

    metric_specs = (
        ("loss", "train_loss", "validation_loss", "test_loss", "Loss"),
        ("auc", "train_auc", "validation_auc", "auc", "AUC"),
        ("precision", "train_precision", "validation_precision", "precision", "Precision"),
        ("recall", "train_recall", "validation_recall", "recall", "Recall"),
        ("accuracy", "train_accuracy", "validation_accuracy", "accuracy", "Accuracy"),
    )
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))

    for axis, (key, train_key, validation_key, test_key, title) in zip(
        axes.flat,
        metric_specs,
    ):
        train_values = [item[train_key] for item in history]
        validation_values = [item[validation_key] for item in history]
        test_value = test_metrics.get(test_key)
        axis.plot(epochs, train_values, marker="o", markersize=3, label="Train")
        axis.plot(
            epochs,
            validation_values,
            marker="o",
            markersize=3,
            label="Validation",
        )
        if test_value is not None and np.isfinite(test_value):
            axis.scatter(
                [epochs[-1]],
                [test_value],
                marker="X",
                s=90,
                label="Test (final)",
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()
        if key != "loss":
            axis.set_ylim(0, 1)

    axes.flat[-1].axis("off")
    figure.suptitle("Training history and final held-out test metrics")
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")

    return figure, axes