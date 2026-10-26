"""SMOTE helpers for continuous DL feature representations."""

import numpy as np
from imblearn.over_sampling import SMOTE


def smote_resample(
    features,
    labels,
    *,
    enabled=True,
    random_state=42,
    k_neighbors=5,
):
    """Apply SMOTE to continuous feature vectors and return X, y, added rows."""
    feature_array = np.asarray(features)
    label_array = np.asarray(labels).reshape(-1)
    if feature_array.ndim != 2 or len(feature_array) != len(label_array):
        raise ValueError("features must be 2D and match the number of labels.")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive.")
    if not enabled:
        return feature_array, label_array, 0

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
    ).fit_resample(feature_array, label_array)
    return (
        np.asarray(resampled_features, dtype=np.float32),
        np.asarray(resampled_labels),
        len(resampled_labels) - original_size,
    )
