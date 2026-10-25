from pathlib import Path
import random
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from DL.CFG import CFG
from DL.dataset import stratified_split_indices
from DL.model import ESM2ForClassification, ProtBertForClassification
import DL.oversampling as oversampling


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


def _build_model(cfg):
    model_classes = {
        "protbert": ProtBertForClassification,
        "esm2": ESM2ForClassification,
    }
    model_type = cfg.MODEL_TYPE.lower()
    if model_type not in model_classes:
        raise ValueError("MODEL_TYPE must be either 'protbert' or 'esm2'.")
    return model_classes[model_type](
        model_name=cfg.MODEL_NAME,
        use_lora=cfg.USE_LORA,
        lora_rank=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        lora_target_modules=cfg.LORA_TARGET_MODULES,
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
    specificity = true_negative / max(negative_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    mcc_denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative)
        / mcc_denominator
        if mcc_denominator
        else 0.0
    )
    pr_auc = (
        float(average_precision_score(targets, probabilities))
        if positive_count
        else float("nan")
    )
    return {
        "auc": float(auc),
        "pr_auc": pr_auc,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": f1,
        "mcc": float(mcc),
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def _save_evaluation_plots(targets, probabilities, threshold, output_dir):
    """Save validation ROC/precision-recall curves and a confusion-matrix heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= threshold
    matrix = np.asarray(
        [
            [
                np.count_nonzero((targets == 0) & ~predictions),
                np.count_nonzero((targets == 0) & predictions),
            ],
            [
                np.count_nonzero((targets == 1) & ~predictions),
                np.count_nonzero((targets == 1) & predictions),
            ],
        ],
        dtype=np.int64,
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, probabilities)
    roc_auc = _binary_metrics([targets], [probabilities], threshold)["auc"]
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
        title="Validation ROC curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(targets, probabilities)
    pr_auc = average_precision_score(targets, probabilities)
    axes[1].plot(recall, precision, label=f"Average precision = {pr_auc:.3f}")
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Validation precision-recall curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[1].legend(loc="best")
    figure.tight_layout()
    curves_path = output_dir / "evaluation_curves.png"
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
        title=f"Validation confusion matrix (threshold={threshold:g})",
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
    matrix_path = output_dir / "confusion_matrix.png"
    figure.savefig(matrix_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return {
        "evaluation_curves": str(curves_path),
        "confusion_matrix": str(matrix_path),
    }


def _save_training_history_plot(history, output_dir):
    """Save per-epoch training and validation loss and classification metrics."""
    if not history:
        raise ValueError("Training history must contain at least one epoch.")
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        epochs,
        [record["train_loss"] for record in history],
        marker="o",
        label="Training",
    )
    if "validation_loss" in history[0]:
        axes[0].plot(
            epochs,
            [record["validation_loss"] for record in history],
            marker="o",
            label="Validation",
        )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Loss",
        title="Training and validation loss",
    )
    axes[0].legend()

    metric_specs = (
        ("ROC-AUC", "auc"),
        ("PR-AUC", "pr_auc"),
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("F1", "f1"),
        ("MCC", "mcc"),
    )
    for label, metric in metric_specs:
        train_key = f"train_{metric}"
        validation_key = f"validation_{metric}"
        if train_key in history[0]:
            axes[1].plot(
                epochs,
                [record[train_key] for record in history],
                marker="o",
                label=f"Training {label}",
            )
        if validation_key in history[0]:
            axes[1].plot(
                epochs,
                [record[validation_key] for record in history],
                marker="o",
                linestyle="--",
                label=f"Validation {label}",
            )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Score",
        title="Training and validation metrics",
    )
    axes[1].legend(ncol=2, fontsize="small")
    figure.tight_layout()
    plot_path = output_dir / "training_validation_history.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(plot_path)


def train_model(
    train_dataset,
    cfg=None,
    model=None,
    device=None,
    checkpoint_path=None,
    seed=None,
    validation_dataset=None,
    fold_number=None,
    n_splits=None,
):
    """Train for fixed epochs and optionally log validation metrics each epoch."""
    cfg = cfg or CFG()
    if cfg.EPOCHS < 1:
        raise ValueError("EPOCHS must be at least 1.")
    if cfg.BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1.")
    if cfg.EARLY_STOPPING_PATIENCE < 1:
        raise ValueError("EARLY_STOPPING_PATIENCE must be at least 1.")
    if cfg.EARLY_STOPPING_MIN_DELTA < 0:
        raise ValueError("EARLY_STOPPING_MIN_DELTA must be non-negative.")

    set_seed(cfg.SEED if seed is None else seed)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if len(train_dataset) == 0:
        raise ValueError("A non-empty train_dataset is required.")
    print("Training configuration:")
    print(f"  Model: {cfg.MODEL_TYPE} ({cfg.MODEL_NAME})")
    print(f"  Dataset: {cfg.DATASET_NAME}")
    print(f"  Training examples: {len(train_dataset):,}")
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
    if validation_dataset is not None:
        print(
            "  Early stopping: validation ROC-AUC, "
            f"patience={cfg.EARLY_STOPPING_PATIENCE}, "
            f"min_delta={cfg.EARLY_STOPPING_MIN_DELTA:g}"
        )

    model = model or _build_model(cfg)
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
            random_state=cfg.SEED if seed is None else seed,
        )
        print(
            "Training-only random oversampling enabled: "
            f"{original_size} -> {len(train_dataset)} examples"
        )
    else:
        print("Training-only random oversampling disabled.")
    train_loader = _make_loader(train_dataset, cfg, shuffle=True, device=device)
    validation_loader = (
        _make_loader(validation_dataset, cfg, shuffle=False, device=device)
        if validation_dataset is not None
        else None
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
    best_validation_auc = float("-inf")
    best_epoch = None
    best_model_state = None
    epochs_without_improvement = 0
    checkpoint_path = Path(checkpoint_path or cfg.CHECKPOINT_PATH)
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
            desc=(
                f"Fold {fold_number}/{n_splits} | Epoch {epoch + 1}/{cfg.EPOCHS}"
                if fold_number is not None
                else f"Epoch {epoch + 1}/{cfg.EPOCHS}"
            ),
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
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_auc": metrics["auc"],
            "train_pr_auc": metrics["pr_auc"],
            "train_recall": metrics["recall"],
            "train_precision": metrics["precision"],
            "train_accuracy": metrics["accuracy"],
            "train_balanced_accuracy": metrics["balanced_accuracy"],
            "train_f1": metrics["f1"],
            "train_mcc": metrics["mcc"],
        }
        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = _evaluate_dataset(
                model=model,
                data_loader=validation_loader,
                criterion=criterion,
                cfg=cfg,
                device=device,
            )
            epoch_record.update(
                {
                    f"validation_{name}": validation_metrics[name]
                    for name in (
                        "loss",
                        "auc",
                        "pr_auc",
                        "recall",
                        "precision",
                        "accuracy",
                        "balanced_accuracy",
                        "f1",
                        "mcc",
                    )
                }
            )
            validation_auc = validation_metrics["auc"]
            if not np.isfinite(validation_auc):
                raise ValueError(
                    "Validation ROC-AUC is undefined; the validation set must "
                    "contain both classes."
                )
            if validation_auc > (
                best_validation_auc + cfg.EARLY_STOPPING_MIN_DELTA
            ):
                best_validation_auc = validation_auc
                best_epoch = epoch + 1
                best_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in base_model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            epoch_record["best_validation_auc"] = best_validation_auc
            epoch_record["early_stopping_wait"] = epochs_without_improvement
        history.append(epoch_record)
        fold_label = (
            f"Fold {fold_number}/{n_splits} | "
            if fold_number is not None
            else ""
        )
        print(
            f"{fold_label}Epoch {epoch + 1}/{cfg.EPOCHS} - "
            f"train_loss: {epoch_loss:.4f}, "
            f"train_auc: {metrics['auc']:.4f}, "
            f"train_recall: {metrics['recall']:.4f}, "
            f"train_precision: {metrics['precision']:.4f}, "
            f"train_accuracy: {metrics['accuracy']:.4f}"
            + (
                ""
                if validation_metrics is None
                else (
                    f", validation_loss: {validation_metrics['loss']:.4f}, "
                    f"validation_auc: {validation_metrics['auc']:.4f}, "
                    f"validation_pr_auc: {validation_metrics['pr_auc']:.4f}, "
                    f"validation_recall: {validation_metrics['recall']:.4f}, "
                    f"validation_precision: {validation_metrics['precision']:.4f}, "
                    f"validation_accuracy: {validation_metrics['accuracy']:.4f}"
                )
            )
        )
        if (
            validation_loader is not None
            and epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping: validation ROC-AUC did not improve for "
                f"{cfg.EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break
    if best_model_state is not None:
        base_model.load_state_dict(best_model_state)
        print(
            f"Restored best model from epoch {best_epoch} "
            f"(validation ROC-AUC={best_validation_auc:.4f})."
        )
    torch.save(
        {
            "model_state_dict": base_model.state_dict(),
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
            "epochs_trained": len(history),
            "best_epoch": best_epoch,
            "best_validation_auc": (
                best_validation_auc if best_epoch is not None else None
            ),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    return model, history


def train_validation_split(
    dataset,
    cfg=None,
    device=None,
    model_factory=None,
):
    """Train once on 80% and evaluate the label-stratified 20% validation set."""
    cfg = cfg or CFG()
    if not hasattr(dataset, "labels"):
        raise TypeError("dataset must expose a labels attribute.")
    labels = dataset.labels.detach().cpu().numpy().reshape(-1)
    train_indices, validation_indices = stratified_split_indices(
        labels,
        validation_size=cfg.VALIDATION_SIZE,
        random_state=cfg.SEED,
    )
    train_subset = Subset(dataset, train_indices.tolist())
    validation_subset = Subset(dataset, validation_indices.tolist())
    print(
        f"Stratified label split: train={len(train_indices):,} "
        f"({len(train_indices) / len(dataset):.0%}), "
        f"validation={len(validation_indices):,} "
        f"({len(validation_indices) / len(dataset):.0%})"
    )
    model, history = train_model(
        train_dataset=train_subset,
        cfg=cfg,
        model=model_factory() if model_factory is not None else None,
        device=device,
        checkpoint_path=cfg.CHECKPOINT_PATH,
        seed=cfg.SEED,
        validation_dataset=validation_subset,
    )
    history_plot = _save_training_history_plot(
        history,
        Path(cfg.CHECKPOINT_PATH).parent,
    )
    evaluation_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    validation_metrics = _evaluate_dataset(
        model=model,
        data_loader=_make_loader(
            validation_subset,
            cfg,
            shuffle=False,
            device=evaluation_device,
        ),
        criterion=_build_loss_function(cfg),
        cfg=cfg,
        device=evaluation_device,
        return_predictions=True,
    )
    validation_targets = validation_metrics.pop("targets")
    validation_probabilities = validation_metrics.pop("probabilities")
    evaluation_plots = _save_evaluation_plots(
        validation_targets,
        validation_probabilities,
        cfg.PREDICTION_THRESHOLD,
        Path(cfg.CHECKPOINT_PATH).parent,
    )
    return {
        "model": model,
        "history": history,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_class_counts": np.bincount(labels[train_indices].astype(int), minlength=2),
        "validation_class_counts": np.bincount(
            labels[validation_indices].astype(int),
            minlength=2,
        ),
        "validation_metrics": validation_metrics,
        "training_history_plot": history_plot,
        "evaluation_plots": evaluation_plots,
    }


def _evaluate_dataset(
    model,
    data_loader,
    criterion,
    cfg,
    device,
    return_predictions=False,
):
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
    results = {
        "loss": total_loss / example_count,
        **metrics,
    }
    if return_predictions:
        results["targets"] = np.concatenate(all_targets)
        results["probabilities"] = np.concatenate(all_probabilities)
    return results


def evaluate_model(model, validation_dataset, cfg=None, device=None):
    """Evaluate a trained model on the held-out validation dataset."""
    cfg = cfg or CFG()
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    validation_loader = _make_loader(
        validation_dataset,
        cfg,
        shuffle=False,
        device=device,
    )
    validation_metrics = _evaluate_dataset(
        model=model,
        data_loader=validation_loader,
        criterion=_build_loss_function(cfg),
        cfg=cfg,
        device=device,
    )
    return {
        f"validation_{name}": value
        for name, value in validation_metrics.items()
    }