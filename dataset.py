from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pre_dataset import preprocess_for_protein_model


class PeptideDataset(Dataset):
    """Tokenized peptide examples compatible with a PyTorch DataLoader."""

    def __init__(
        self,
        encodings,
    ):
        self.encodings = {
            key: torch.tensor(values, dtype=torch.long)
            for key, values in encodings.items()
            if key != "labels"
        }
        self.labels = torch.tensor(encodings["labels"], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def prepare_dataset(
    dataset_name,
    tokenizer,
    data_dir="dataset",
    max_length=32,
    model_type="protbert",
):
    """Tokenize the complete cleaned dataset for cross-validation."""
    if model_type.lower() not in {"protbert", "esm2"}:
        raise ValueError("model_type must be either 'protbert' or 'esm2'.")
    dataframe = prepare_dataset_frames(
        dataset_name=dataset_name,
        data_dir=data_dir,
    )
    encodings = preprocess_for_protein_model(
        dataframe,
        tokenizer,
        model_type=model_type,
        max_length=max_length,
        sequence_column="peptide_seq",
    )
    return PeptideDataset(encodings)


def prepare_dataset_frames(
    dataset_name,
    data_dir="dataset",
):
    """Clean the selected peptide CSV and return all rows for cross-validation."""
    filenames = {
        "bcell": "input_bcell.csv",
        "sars": "input_sars.csv",
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in filenames:
        raise ValueError("dataset_name must be either 'bcell' or 'sars'.")

    csv_path = Path(data_dir) / filenames[dataset_name]
    dataframe = pd.read_csv(csv_path)
    required_columns = {"peptide_seq", "target"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

    dataframe = dataframe.reset_index(drop=True)
    labels = pd.to_numeric(dataframe["target"], errors="raise")
    if not labels.isin((0, 1)).all():
        raise ValueError("'target' must contain only binary labels 0 and 1.")
    dataframe["target"] = labels.astype("int64")
    dataframe, cleaning_summary = _clean_dataset(dataframe, dataset_name)
    print(
        f"Dataset cleaning ({dataset_name}): "
        f"{cleaning_summary['input_rows']:,} -> {len(dataframe):,} rows; "
        f"removed {cleaning_summary['conflicting_label_rows']:,} rows from "
        f"{cleaning_summary['conflicting_label_sequences']:,} contradictory "
        "B-cell peptide sequences, "
        f"{cleaning_summary['long_sequence_rows']:,} SARS rows longer than "
        "20 residues, and "
        f"{cleaning_summary['duplicate_sequence_rows']:,} duplicate peptide "
        "sequences."
    )
    if dataframe.empty or dataframe["target"].nunique() != 2:
        raise ValueError("The cleaned dataset must contain examples of both classes.")
    return dataframe


def stratified_kfold_indices(labels, n_splits=5, random_state=42):
    """Return deterministic row-level folds with both classes represented."""
    labels = np.asarray(labels).reshape(-1)
    if (
        not isinstance(n_splits, (int, np.integer))
        or isinstance(n_splits, (bool, np.bool_))
        or n_splits < 2
    ):
        raise ValueError("n_splits must be at least 2.")
    if labels.size == 0:
        raise ValueError("Cannot split an empty dataset.")

    classes, encoded_labels, class_counts = np.unique(
        labels,
        return_inverse=True,
        return_counts=True,
    )
    if len(classes) != 2:
        raise ValueError("Stratified binary folds require exactly two classes.")
    if class_counts.min() < n_splits:
        raise ValueError(
            "Each class must contain at least n_splits examples for "
            "stratified k-fold."
        )

    rng = np.random.default_rng(random_state)
    encoded_order = np.sort(encoded_labels)
    fold_allocation = np.asarray(
        [
            np.bincount(encoded_order[fold_index::n_splits], minlength=len(classes))
            for fold_index in range(n_splits)
        ]
    )
    fold_assignments = np.empty(labels.size, dtype=np.int64)
    for class_index, fold_counts in enumerate(fold_allocation.T):
        class_indices = np.flatnonzero(encoded_labels == class_index)
        rng.shuffle(class_indices)
        start = 0
        for fold_index, count in enumerate(fold_counts):
            end = start + int(count)
            fold_assignments[class_indices[start:end]] = fold_index
            start = end

    all_indices = np.arange(labels.size)
    folds = []
    for fold_index in range(n_splits):
        validation_indices = np.flatnonzero(fold_assignments == fold_index)
        training_mask = np.ones(labels.size, dtype=bool)
        training_mask[validation_indices] = False
        folds.append((all_indices[training_mask], validation_indices))
    return tuple(folds)


def _clean_dataset(dataframe, dataset_name):
    """Remove contradictory, overlength, and duplicate peptide examples."""
    dataframe = dataframe.copy()
    normalized_sequences = (
        dataframe["peptide_seq"]
        .astype("string")
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )
    if normalized_sequences.isna().any() or normalized_sequences.eq("").any():
        raise ValueError("'peptide_seq' must not contain missing or empty values.")

    dataframe["_normalized_peptide_seq"] = normalized_sequences
    conflicting_label_rows = 0
    conflicting_label_sequences = 0
    long_sequence_rows = 0

    if dataset_name == "bcell":
        conflicting = dataframe.groupby("_normalized_peptide_seq")[
            "target"
        ].transform("nunique").gt(1)
        conflicting_label_rows = int(conflicting.sum())
        conflicting_label_sequences = int(
            dataframe.loc[conflicting, "_normalized_peptide_seq"].nunique()
        )
        dataframe = dataframe.loc[~conflicting]
    else:
        overlength = dataframe["_normalized_peptide_seq"].str.len().gt(20)
        long_sequence_rows = int(overlength.sum())
        dataframe = dataframe.loc[~overlength]

    duplicate_sequences = dataframe["_normalized_peptide_seq"].duplicated(
        keep="first"
    )
    duplicate_sequence_rows = int(duplicate_sequences.sum())
    dataframe = (
        dataframe.loc[~duplicate_sequences]
        .drop(columns="_normalized_peptide_seq")
        .reset_index(drop=True)
    )

    return dataframe, {
        "input_rows": len(normalized_sequences),
        "conflicting_label_rows": conflicting_label_rows,
        "conflicting_label_sequences": conflicting_label_sequences,
        "long_sequence_rows": long_sequence_rows,
        "duplicate_sequence_rows": duplicate_sequence_rows,
    }
