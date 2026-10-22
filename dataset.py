from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pre_dataset import preprocess_for_protein_model


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
    train_size=0.7,
    validation_size=0.15,
    test_size=0.15,
    random_state=42,
    max_length=32,
    model_type="protbert",
):
    """Tokenize peptide_seq and return train/validation/test datasets.

    B-cell examples are split by parent protein to prevent protein-level leakage.
    Identical normalized SARS peptide sequences are kept in the same split.
    protein_seq is metadata and is not passed to the language model.
    """
    filenames = {
        "bcell": "input_bcell.csv",
        "sars": "input_sars.csv",
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in filenames:
        raise ValueError("dataset_name must be either 'bcell' or 'sars'.")
    if model_type.lower() not in {"protbert", "esm2"}:
        raise ValueError("model_type must be either 'protbert' or 'esm2'.")
    split_sizes = np.asarray([train_size, validation_size, test_size], dtype=float)
    if (
        not np.isfinite(split_sizes).all()
        or (split_sizes <= 0).any()
        or not np.isclose(split_sizes.sum(), 1.0)
    ):
        raise ValueError(
            "train_size, validation_size, and test_size must be positive "
            "and sum to 1."
        )

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
        group_ids = dataframe["parent_protein_id"]
    else:
        normalized_sequences = (
            dataframe["peptide_seq"]
            .astype("string")
            .str.upper()
            .str.replace(r"\s+", "", regex=True)
        )
        if normalized_sequences.isna().any() or normalized_sequences.eq("").any():
            raise ValueError("'peptide_seq' must not contain missing or empty values.")
        group_ids = normalized_sequences

    frames = _grouped_train_validation_test_split(
        dataframe=dataframe,
        group_ids=group_ids,
        split_sizes=split_sizes,
        rng=rng,
    )
    train_frame, validation_frame, test_frame = frames

    for split_name, split_frame in zip(
        ("training", "validation", "test"),
        frames,
    ):
        if split_frame.empty or split_frame["target"].nunique() != 2:
            raise ValueError(
                f"The {split_name} split does not contain both target classes."
            )

    datasets = []
    for split_frame in frames:
        encodings = preprocess_for_protein_model(
            split_frame,
            tokenizer,
            model_type=model_type,
            max_length=max_length,
            sequence_column="peptide_seq",
        )
        datasets.append(PeptideDataset(encodings))
    return tuple(datasets)


def _grouped_train_validation_test_split(dataframe, group_ids, split_sizes, rng):
    """Assign whole groups to approximate the requested sizes and class ratios."""
    group_frame = pd.DataFrame(
        {
            "group": group_ids.to_numpy(),
            "target": dataframe["target"].to_numpy(),
        }
    )
    group_counts = group_frame.groupby("group", sort=False)["target"].agg(
        total="size",
        positive="sum",
    )
    group_counts["negative"] = group_counts["total"] - group_counts["positive"]
    groups = group_counts.index.to_numpy()
    counts = group_counts[["total", "positive", "negative"]].to_numpy()
    if len(groups) < 3:
        raise ValueError("A three-way grouped split requires at least three groups.")

    total_counts = counts.sum(axis=0)
    target_counts = split_sizes[:, None] * total_counts
    denominators = np.maximum(target_counts, 1)
    best_score = float("inf")
    best_splits = None

    for _ in range(512):
        order = rng.permutation(len(groups))
        ordered_counts = counts[order]
        cumulative = np.vstack(
            (np.zeros(3, dtype=np.int64), np.cumsum(ordered_counts, axis=0))
        )

        train_errors = np.square(
            (cumulative - target_counts[0]) / denominators[0]
        ).sum(axis=1)
        train_errors[[0, -2, -1]] = np.inf
        train_end = int(train_errors.argmin())

        remaining_counts = cumulative[-1] - cumulative[train_end]
        val_prefixes = cumulative[train_end + 1 : -1] - cumulative[train_end]
        validation_errors = np.square(
            (val_prefixes - target_counts[1]) / denominators[1]
        ).sum(axis=1)
        if len(validation_errors) == 0:
            continue

        for validation_offset in np.argsort(validation_errors)[:8]:
            validation_end = train_end + 1 + int(validation_offset)
            train_counts = cumulative[train_end]
            validation_counts = cumulative[validation_end] - train_counts
            test_counts = remaining_counts - validation_counts
            candidate_counts = (train_counts, validation_counts, test_counts)
            if any((split_counts[1:] == 0).any() for split_counts in candidate_counts):
                continue

            score = sum(
                np.square((observed - target) / denominator).sum()
                for observed, target, denominator in zip(
                    candidate_counts,
                    target_counts,
                    denominators,
                )
            )
            if score < best_score:
                best_score = score
                train_groups = groups[order[:train_end]]
                validation_groups = groups[order[train_end:validation_end]]
                test_groups = groups[order[validation_end:]]
                best_splits = (train_groups, validation_groups, test_groups)

    if best_splits is None:
        raise ValueError(
            "Could not create three non-empty grouped splits containing both "
            "target classes. More groups per class may be required."
        )

    return tuple(
        dataframe.loc[group_ids.isin(split_groups)].copy()
        for split_groups in best_splits
    )