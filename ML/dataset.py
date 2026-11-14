"""Loading, validating, and cleaning labeled peptide datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from ML.pre_dataset import get_bert_cls_columns, normalize_sequence


DATASET_FILENAMES = {
    "bcell": "input_bcell.csv",
    "sars": "input_sars.csv",
}
PROTTRANS_MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
PROTTRANS_TOKENIZER_NAME = "Rostlab/prot_t5_xl_uniref50"
PROTTRANS_DIMENSION = 1024


def _read_prottrans_embeddings(
    dataset_name: str,
    data_dir: str | Path,
    embedding_dir: str | Path | None,
) -> tuple[list[str], np.ndarray]:
    """Read precomputed ProtT5 mean vectors; never generate sequence embeddings."""
    if embedding_dir is not None:
        cache_dir = Path(embedding_dir)
        cache_path = cache_dir / f"{dataset_name}_prott5.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Precomputed BLMPred vectors not found: {cache_path}. "
                "Expected a NumPy .npz with sequences, embeddings, model_name, "
                "and tokenizer_name arrays."
            )
        with np.load(cache_path, allow_pickle=False) as cache:
            required = {"sequences", "embeddings", "model_name", "tokenizer_name"}
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(
                    f"{cache_path.name} is missing arrays: {sorted(missing)}."
                )
            sequences = cache["sequences"].astype(str).tolist()
            embeddings = cache["embeddings"].astype(np.float32)
            model_name = str(cache["model_name"].item())
            tokenizer_name = str(cache["tokenizer_name"].item())
    elif dataset_name == "bcell":
        # Reuse the existing frozen ProtT5 cache created by EmbeddingML.
        cache_dir = Path(__file__).resolve().parents[1] / "EmbeddingML" / "artifacts" / "RF"
        required_files = (
            "sequences.npy",
            "embeddings.npy",
            "model_name.npy",
            "tokenizer_name.npy",
        )
        missing_files = [name for name in required_files if not (cache_dir / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                "The default BLMPred B-cell ProtT5 cache is incomplete at "
                f"{cache_dir}; missing {missing_files}. Set "
                "PROTTRANS_EMBEDDING_DIR to a directory containing "
                "bcell_prott5.npz and sars_prott5.npz."
            )
        sequences = np.load(cache_dir / "sequences.npy", allow_pickle=False).astype(str).tolist()
        embeddings = np.load(cache_dir / "embeddings.npy", allow_pickle=False).astype(np.float32)
        model_name = str(np.load(cache_dir / "model_name.npy", allow_pickle=False).item())
        tokenizer_name = str(np.load(cache_dir / "tokenizer_name.npy", allow_pickle=False).item())
    else:
        cache_path = Path(data_dir) / "prott5_embeddings" / f"{dataset_name}_prott5.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(
                "Precomputed SARS ProtT5 vectors are required for perbandingan_5. "
                f"Expected {cache_path}, or set PROTTRANS_EMBEDDING_DIR to a "
                "directory containing sars_prott5.npz."
            )
        with np.load(cache_path, allow_pickle=False) as cache:
            required = {"sequences", "embeddings", "model_name", "tokenizer_name"}
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(
                    f"{cache_path.name} is missing arrays: {sorted(missing)}."
                )
            sequences = cache["sequences"].astype(str).tolist()
            embeddings = cache["embeddings"].astype(np.float32)
            model_name = str(cache["model_name"].item())
            tokenizer_name = str(cache["tokenizer_name"].item())

    if model_name != PROTTRANS_MODEL_NAME or tokenizer_name != PROTTRANS_TOKENIZER_NAME:
        raise ValueError(
            "perbandingan_5 requires the ProtT5-XL-U50 encoder and tokenizer "
            f"({PROTTRANS_MODEL_NAME!r}, {PROTTRANS_TOKENIZER_NAME!r}); got "
            f"{model_name!r}, {tokenizer_name!r}."
        )
    if embeddings.ndim != 2 or embeddings.shape != (len(sequences), PROTTRANS_DIMENSION):
        raise ValueError(
            "ProtT5 embeddings must have shape "
            f"(number_of_sequences, {PROTTRANS_DIMENSION}); got {embeddings.shape}."
        )
    if len(set(sequences)) != len(sequences):
        raise ValueError("ProtT5 cache must contain one row per unique peptide.")
    if not np.isfinite(embeddings).all():
        raise ValueError("ProtT5 cache contains non-finite values.")
    return sequences, embeddings


def load_labeled_dataset(
    dataset_name: str = "bcell",
    data_dir: str | Path = "dataset",
    max_peptide_length: int = 20,
    feature_set: str = "proposed",
    bert_embedding_dir: str | Path | None = None,
    prottrans_embedding_dir: str | Path | None = None,
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

    if feature_set == "perbandingan_4":
        bert_columns = get_bert_cls_columns(dataframe.columns)
        if not bert_columns:
            embedding_dir = (
                Path(bert_embedding_dir)
                if bert_embedding_dir is not None
                else Path(data_dir) / "bert_embeddings"
            )
            embedding_path = embedding_dir / f"{dataset_name}_bert_cls.csv"
            if not embedding_path.is_file():
                raise FileNotFoundError(
                    "perbandingan_4 requires precomputed CLS vectors. Put a CSV "
                    f"with peptide_seq and bert_cls_000... columns at {embedding_path}."
                )
            embeddings = pd.read_csv(embedding_path)
            if "peptide_seq" not in embeddings.columns:
                raise ValueError(
                    f"{embedding_path.name} must contain a peptide_seq key column."
                )
            bert_columns = get_bert_cls_columns(embeddings.columns)
            if not bert_columns:
                raise ValueError(
                    f"{embedding_path.name} has no bert_cls_000, bert_cls_001, ... columns."
                )
            embeddings = embeddings.loc[:, ["peptide_seq", *bert_columns]].copy()
            embeddings["peptide_seq"] = [
                normalize_sequence(value, row_label=index)
                for index, value in embeddings["peptide_seq"].items()
            ]
            if embeddings["peptide_seq"].duplicated().any():
                raise ValueError(
                    f"{embedding_path.name} must have one row per unique peptide_seq."
                )
            for column in bert_columns:
                embeddings[column] = pd.to_numeric(
                    embeddings[column], errors="raise"
                )
            if not np.isfinite(embeddings.loc[:, bert_columns].to_numpy(dtype=float)).all():
                raise ValueError(
                    f"{embedding_path.name} contains non-finite CLS feature values."
                )
            dataframe = dataframe.merge(
                embeddings,
                on="peptide_seq",
                how="left",
                validate="many_to_one",
                sort=False,
            )
            missing_embedding = dataframe[list(bert_columns)].isna().any(axis=1)
            if missing_embedding.any():
                missing_sequences = dataframe.loc[
                    missing_embedding, "peptide_seq"
                ].drop_duplicates()
                raise ValueError(
                    f"{embedding_path.name} is missing CLS vectors for "
                    f"{len(missing_sequences):,} cleaned dataset peptide(s)."
                )
        elif len(bert_columns) != len(get_bert_cls_columns(dataframe.columns)):
            raise ValueError("Inconsistent BERT CLS feature columns in dataset.")

    if feature_set == "perbandingan_5":
        sequences, embeddings = _read_prottrans_embeddings(
            dataset_name,
            data_dir,
            prottrans_embedding_dir,
        )
        embedding_columns = [f"prott5_{index:04d}" for index in range(embeddings.shape[1])]
        embedding_frame = pd.DataFrame(embeddings, columns=embedding_columns)
        embedding_frame.insert(0, "peptide_seq", [
            normalize_sequence(value, row_label=index)
            for index, value in enumerate(sequences)
        ])
        if embedding_frame["peptide_seq"].duplicated().any():
            raise ValueError("ProtT5 cache has duplicate normalized peptide keys.")
        dataframe = dataframe.merge(
            embedding_frame,
            on="peptide_seq",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        missing_embedding = dataframe[embedding_columns].isna().any(axis=1)
        if missing_embedding.any():
            missing_count = int(dataframe.loc[missing_embedding, "peptide_seq"].nunique())
            raise ValueError(
                f"ProtT5 cache is missing vectors for {missing_count:,} cleaned "
                f"{dataset_name} peptide(s)."
            )

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


def make_cross_validation_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    dataset_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Create five stratified folds, grouping proteins except for SARS."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if labels.size == 0 or labels.size != groups.size:
        raise ValueError("labels and groups must have the same non-zero length.")
    if np.unique(labels).size != 2:
        raise ValueError("Cross-validation requires examples of both classes.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if dataset_name.lower() == "sars":
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        folds = list(splitter.split(np.zeros(labels.size), labels))
        strategy = "stratified_peptide_5_fold"
    else:
        if pd.isna(groups).any():
            raise ValueError("Protein groups must not contain missing values.")
        if pd.unique(groups).size < n_splits:
            raise ValueError(
                f"Protein-level {n_splits}-fold CV requires at least "
                f"{n_splits} proteins; found {pd.unique(groups).size}."
            )
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        folds = list(splitter.split(np.zeros(labels.size), labels, groups))
        strategy = "stratified_protein_group_5_fold"

    for fold_number, (train_indices, validation_indices) in enumerate(folds, 1):
        if (
            np.unique(labels[train_indices]).size != 2
            or np.unique(labels[validation_indices]).size != 2
        ):
            raise ValueError(
                f"{strategy} fold {fold_number} does not contain both classes "
                "in train and validation."
            )
        if dataset_name.lower() != "sars" and np.intersect1d(
            groups[train_indices],
            groups[validation_indices],
        ).size:
            raise RuntimeError(
                f"Protein overlap detected in cross-validation fold {fold_number}."
            )
    return folds, strategy
