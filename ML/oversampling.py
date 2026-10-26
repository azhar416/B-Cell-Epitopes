"""SMOTE helpers for numeric ML feature matrices."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def smote_resample(
    features,
    labels,
    *,
    enabled=True,
    random_state=42,
    k_neighbors=5,
):
    """Apply SMOTE to numeric features and return X, y, added row count."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    label_array = np.asarray(labels).reshape(-1)
    if len(features) != len(label_array):
        raise ValueError("features and labels must contain the same number of rows.")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive.")
    if not enabled:
        return features.copy(), label_array, 0

    class_counts = np.bincount(label_array.astype(int), minlength=2)
    if class_counts.min() <= k_neighbors:
        raise ValueError(
            "k_neighbors must be smaller than the minority-class count "
            f"({class_counts.min()})."
        )
    original_size = len(label_array)
    resampled_features, resampled_labels = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
    ).fit_resample(features, label_array)
    return (
        pd.DataFrame(resampled_features, columns=features.columns),
        np.asarray(resampled_labels),
        len(resampled_labels) - original_size,
    )