from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pre_dataset import preprocess_for_protbert


class PeptideDataset(Dataset):
    """Tokenized peptide examples compatible with a PyTorch DataLoader."""

    def __init__(self, encodings):
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
    test_size=0.2,
    random_state=42,
    max_length=32,
):
    """Tokenize peptide_seq and return train/test PyTorch datasets.

    B-cell examples are split by parent protein to prevent protein-level leakage.
    The SARS dataset contains one parent protein, so it is stratified by label
    at the row level instead. protein_seq is metadata and is not passed to ProtBert.
    """
    filenames = {
        "bcell": "input_bcell.csv",
        "sars": "input_sars.csv",
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in filenames:
        raise ValueError("dataset_name must be either 'bcell' or 'sars'.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

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
    rng = np.random.default_rng(random_state)

    if dataset_name == "bcell":
        if "parent_protein_id" not in dataframe:
            raise ValueError(f"{csv_path.name} is missing 'parent_protein_id'.")
        if dataframe["parent_protein_id"].isna().any():
            raise ValueError("'parent_protein_id' must not contain missing values.")

        group_counts = dataframe.groupby("parent_protein_id")["target"].agg(
            total="size",
            positive="sum",
        )
        group_counts["negative"] = group_counts["total"] - group_counts["positive"]
        groups = group_counts.index.to_numpy()
        if len(groups) < 2:
            raise ValueError("A protein-grouped split requires at least two proteins.")
        counts = group_counts[["total", "positive", "negative"]].to_numpy()
        target_counts = counts.sum(axis=0) * test_size
        denominators = np.maximum(target_counts, 1)
        best_score = float("inf")
        best_test_groups = groups[:0]

        for _ in range(128):
            group_order = rng.permutation(len(groups))
            cumulative_counts = np.vstack(
                (np.zeros(3, dtype=np.int64), np.cumsum(counts[group_order], axis=0))
            )
            relative_errors = (cumulative_counts - target_counts) / denominators
            scores = np.square(relative_errors).sum(axis=1)
            scores[[0, -1]] = np.inf
            best_count = int(scores.argmin())
            if scores[best_count] < best_score:
                best_score = scores[best_count]
                best_test_groups = groups[group_order[:best_count]]

        test_groups = set(best_test_groups)
        test_mask = dataframe["parent_protein_id"].isin(test_groups)
        train_frame = dataframe.loc[~test_mask].copy()
        test_frame = dataframe.loc[test_mask].copy()
    else:
        test_indices = []
        for label, label_frame in dataframe.groupby("target", sort=True):
            if len(label_frame) < 2:
                raise ValueError(
                    f"Cannot stratify label {label}: at least two rows are required."
                )
            indices = label_frame.index.to_numpy(copy=True)
            rng.shuffle(indices)
            label_test_count = min(
                max(1, round(len(indices) * test_size)),
                len(indices) - 1,
            )
            test_indices.extend(indices[:label_test_count])

        test_mask = dataframe.index.isin(test_indices)
        train_frame = dataframe.loc[~test_mask].copy()
        test_frame = dataframe.loc[test_mask].copy()

    for split_name, split_frame in (("training", train_frame), ("test", test_frame)):
        if split_frame.empty or split_frame["target"].nunique() != 2:
            raise ValueError(
                f"The {split_name} split does not contain both target classes."
            )

    train_encodings = preprocess_for_protbert(
        train_frame,
        tokenizer,
        max_length=max_length,
        sequence_column="peptide_seq",
    )
    test_encodings = preprocess_for_protbert(
        test_frame,
        tokenizer,
        max_length=max_length,
        sequence_column="peptide_seq",
    )
    return PeptideDataset(train_encodings), PeptideDataset(test_encodings)