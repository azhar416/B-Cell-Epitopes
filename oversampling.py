import numpy as np
from torch.utils.data import Subset


USE_OVERSAMPLING = False  # Set to False to disable oversampling of the training dataset


def oversample_training_dataset(dataset, random_state=42, enabled=None):
    """Randomly duplicate minority-class training examples to balance classes.

    This is random oversampling, not SMOTE: interpolating token IDs would not
    produce valid protein sequences. Validation and test datasets must not be
    passed to this function.
    """
    if enabled is None:
        enabled = USE_OVERSAMPLING
    if not enabled:
        return dataset

    labels = dataset.labels.detach().cpu().numpy().reshape(-1)
    classes, class_counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError(
            "Oversampling requires a training dataset containing both classes."
        )

    majority_count = int(class_counts.max())
    rng = np.random.default_rng(random_state)
    balanced_indices = []
    for class_label, class_count in zip(classes, class_counts):
        class_indices = np.flatnonzero(labels == class_label)
        extra_count = majority_count - int(class_count)
        if extra_count:
            extra_indices = rng.choice(
                class_indices,
                size=extra_count,
                replace=True,
            )
            class_indices = np.concatenate((class_indices, extra_indices))
        balanced_indices.append(class_indices)

    indices = np.concatenate(balanced_indices)
    rng.shuffle(indices)
    return Subset(dataset, indices.tolist())