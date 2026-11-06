"""Loading, validating, and cleaning labeled peptide datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from ML.pre_dataset import normalize_sequence


DATASET_FILENAMES = {
    "bcell": "input_bcell.csv",
    "sars": "input_sars.csv",
}


def load_labeled_dataset(
    dataset_name: str = "bcell",
    data_dir: str | Path = "dataset",
    max_peptide_length: int = 20,
) -> pd.DataFrame:
    """Load and clean a labeled CSV, retaining protein IDs for grouped splitting."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_FILENAMES:
        raise ValueError(
            "dataset_name must be 'bcell' or 'sars'; input_covid.csv has no labels."
        )
    if max_peptide_length < 1:
        raise ValueError("max_peptide_length must be positive.")

    csv_path = Path(data_dir) / DATASET_FILENAMES[dataset_name]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    dataframe = pd.read_csv(csv_path)
    required_columns = {"peptide_seq", "target", "parent_protein_id"}
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError(f"{csv_path.name} contains no rows.")

    dataframe = dataframe.copy().reset_index(drop=True)
    dataframe["peptide_seq"] = [
        normalize_sequence(value, row_label=index)
        for index, value in dataframe["peptide_seq"].items()
    ]
    if dataframe["parent_protein_id"].isna().any():
        raise ValueError("'parent_protein_id' must not contain missing values.")

    labels = pd.to_numeric(dataframe["target"], errors="raise")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("'target' must contain only binary labels 0 and 1.")
    dataframe["target"] = labels.astype("int64")

    input_rows = len(dataframe)
    long_sequences = dataframe["peptide_seq"].str.len().gt(max_peptide_length)
    removed_long = int(long_sequences.sum())
    dataframe = dataframe.loc[~long_sequences].copy()

    contradictory = dataframe.groupby("peptide_seq")["target"].transform(
        "nunique"
    ).gt(1)
    removed_conflicting = int(contradictory.sum())
    dataframe = dataframe.loc[~contradictory].copy()

    duplicate = dataframe["peptide_seq"].duplicated(keep="first")
    removed_duplicates = int(duplicate.sum())
    dataframe = dataframe.loc[~duplicate].reset_index(drop=True)

    if dataframe.empty or dataframe["target"].nunique() != 2:
        raise ValueError("The cleaned dataset must contain examples of both classes.")

    print(
        f"Cleaning {dataset_name}: {input_rows:,} -> {len(dataframe):,} rows "
        f"(removed {removed_long:,} overlength, "
        f"{removed_conflicting:,} conflicting-label, "
        f"{removed_duplicates:,} duplicate rows)."
    )
    print(
        "Class counts: "
        f"{dataframe['target'].value_counts().sort_index().to_dict()}; "
        f"parent proteins: {dataframe['parent_protein_id'].nunique():,}."
    )
    return dataframe


def make_stratified_split(
    labels: np.ndarray,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the legacy row-level split stratified by label only."""
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("A stratified split requires non-empty binary labels.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    class_counts = np.bincount(labels.astype(int), minlength=2)
    validation_counts = np.ceil(class_counts * validation_size).astype(int)
    if class_counts.min() < 2 or validation_counts.min() < 1:
        raise ValueError(
            "Each class must have enough examples to appear in both train and "
            "validation sets."
        )

    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)),
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )
    if np.unique(labels[train_indices]).size != 2 or np.unique(
        labels[validation_indices]
    ).size != 2:
        raise ValueError("Both classes must be present in train and validation sets.")
    return train_indices, validation_indices


def make_grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by protein while approximating the requested size and class ratio."""
    labels = np.asarray(labels).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if labels.size == 0 or labels.size != groups.size:
        raise ValueError("labels and groups must have the same non-zero length.")
    if np.unique(labels).size != 2:
        raise ValueError("A grouped split requires examples of both classes.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if pd.isna(groups).any():
        raise ValueError("Protein groups must not contain missing values.")
    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            "A protein-level holdout requires at least two distinct proteins; "
            f"found {len(unique_groups)}."
        )

    overall_positive_rate = float(np.mean(labels))
    best_split = None
    best_score = float("inf")
    splitter = GroupShuffleSplit(
        n_splits=256,
        test_size=validation_size,
        random_state=random_state,
    )
    for train_indices, validation_indices in splitter.split(
        np.zeros(labels.size),
        labels,
        groups,
    ):
        if (
            np.unique(labels[train_indices]).size != 2
            or np.unique(labels[validation_indices]).size != 2
        ):
            continue
        actual_validation_size = len(validation_indices) / labels.size
        validation_positive_rate = float(np.mean(labels[validation_indices]))
        score = (
            abs(actual_validation_size - validation_size) / validation_size
            + abs(validation_positive_rate - overall_positive_rate)
        )
        if score < best_score:
            best_score = score
            best_split = train_indices, validation_indices

    if best_split is None:
        raise ValueError(
            "Could not create a protein-level split containing both classes "
            "in training and validation. More proteins may be required."
        )
    train_indices, validation_indices = best_split
    if np.intersect1d(groups[train_indices], groups[validation_indices]).size:
        raise RuntimeError("Protein-level split placed a group in both partitions.")
    return train_indices, validation_indices