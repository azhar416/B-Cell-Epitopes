"""Dataset splitting and input feature assembly."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from EmbeddingML.preprocessing import BIOLOGICAL_FEATURE_COLUMNS


def make_stratified_split(
    labels: np.ndarray,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("A stratified split requires examples of both classes.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    return train_test_split(
        np.arange(labels.size),
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )


def make_grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out complete proteins, choosing a deterministic near-80:20 split."""
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
    if pd.unique(groups).size < 2:
        raise ValueError(
            "A protein-level holdout requires at least two distinct proteins."
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
        size_error = abs(
            len(validation_indices) / labels.size - validation_size
        ) / validation_size
        prevalence_error = abs(
            float(np.mean(labels[validation_indices])) - overall_positive_rate
        )
        score = size_error + prevalence_error
        if score < best_score:
            best_score = score
            best_split = train_indices, validation_indices
    if best_split is None:
        raise ValueError(
            "Could not create a protein-level split containing both classes "
            "in both partitions."
        )

    train_indices, validation_indices = best_split
    if np.intersect1d(groups[train_indices], groups[validation_indices]).size:
        raise RuntimeError("A protein was assigned to both split partitions.")
    return train_indices, validation_indices


def make_dataset_split(
    dataframe: pd.DataFrame,
    dataset_name: str,
    validation_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    labels = dataframe["target"].to_numpy(dtype=np.int64)
    groups = dataframe["parent_protein_id"].to_numpy()
    if dataset_name.lower() == "sars" and pd.unique(groups).size == 1:
        train_indices, validation_indices = make_stratified_split(
            labels,
            validation_size=validation_size,
            random_state=random_state,
        )
        return (
            train_indices,
            validation_indices,
            "stratified_peptide_single_protein",
        )
    train_indices, validation_indices = make_grouped_split(
        labels,
        groups,
        validation_size=validation_size,
        random_state=random_state,
    )
    return train_indices, validation_indices, "protein_group_holdout"


def assemble_features(
    embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    use_biological_features: bool = False,
) -> tuple[np.ndarray, list[str]]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(dataframe):
        raise ValueError("Embeddings must be a 2D array aligned with the dataset.")
    feature_names = [f"prott5_{index}" for index in range(embeddings.shape[1])]
    if not use_biological_features:
        return embeddings, feature_names
    descriptors = dataframe[list(BIOLOGICAL_FEATURE_COLUMNS)].to_numpy(
        dtype=np.float32
    )
    return (
        np.concatenate((embeddings, descriptors), axis=1),
        feature_names + list(BIOLOGICAL_FEATURE_COLUMNS),
    )