import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from TF.pre_dataset import (
    extract_flanked_sequence,
    preprocess_for_protein_model,
    resolve_max_length,
)


DATASET_FILENAMES = {
    "bcell": "input_bcell.csv",
    "sars": "input_sars.csv",
}
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
RARE_AMINO_ACIDS = frozenset("BUZO")
DESCRIPTOR_COLUMNS = (
    "chou_fasman",
    "emini",
    "kolaskar_tongaonkar",
    "parker",
    "isoelectric_point",
    "aromaticity",
    "hydrophobicity",
    "stability",
)


def _normalize_sequence(value, row_label):
    if pd.isna(value):
        raise ValueError(f"Missing peptide sequence at row {row_label}.")
    sequence = "".join(str(value).upper().split())
    if not sequence:
        raise ValueError(f"Empty peptide sequence at row {row_label}.")
    invalid = set(sequence).difference(
        CANONICAL_AMINO_ACIDS | RARE_AMINO_ACIDS | {"X"}
    )
    if invalid:
        raise ValueError(
            f"Unsupported residues at row {row_label}: {sorted(invalid)}."
        )
    return "".join("X" if residue in RARE_AMINO_ACIDS else residue for residue in sequence)


class PeptideDataset(Dataset):
    """Tokenized peptide examples compatible with a PyTorch DataLoader."""

    def __init__(
        self,
        encodings,
        group_ids=None,
        descriptor_features=None,
    ):
        self.encodings = {
            key: torch.tensor(values, dtype=torch.long)
            for key, values in encodings.items()
            if key != "labels"
        }
        self.labels = torch.tensor(encodings["labels"], dtype=torch.float32)
        self.descriptor_features = (
            torch.tensor(descriptor_features, dtype=torch.float32)
            if descriptor_features is not None
            else None
        )
        self.group_ids = (
            np.asarray(group_ids) if group_ids is not None else None
        )
        if self.group_ids is not None and len(self.group_ids) != len(self.labels):
            raise ValueError("group_ids must contain one parent ID per example.")
        if (
            self.descriptor_features is not None
            and len(self.descriptor_features) != len(self.labels)
        ):
            raise ValueError(
                "descriptor_features must contain one row per example."
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}
        if self.descriptor_features is not None:
            item["features"] = self.descriptor_features[index]
        item["labels"] = self.labels[index]
        return item


def prepare_dataset(
    dataset_name,
    tokenizer,
    data_dir="dataset",
    max_length=None,
    model_type="protbert",
    max_peptide_length=20,
    use_protein_features=False,
    flanking_size=5,
    cfg=None,
):
    """Tokenize the cleaned data with optional flanking protein context."""
    if cfg is not None:
        dataset_name = getattr(cfg, "DATASET_NAME", dataset_name)
        data_dir = getattr(cfg, "_DATASET_PATH", data_dir)
        model_type = getattr(cfg, "MODEL_TYPE", model_type)
        max_peptide_length = getattr(cfg, "MAX_PEPTIDE_LENGTH", max_peptide_length)
        use_protein_features = getattr(cfg, "USE_PROTEIN_FEATURES", use_protein_features)
        flanking_size = getattr(cfg, "FLANKING_SIZE", flanking_size)

    if model_type.lower() not in {"protbert", "esm2"}:
        raise ValueError("model_type must be either 'protbert' or 'esm2'.")
    if flanking_size < 0:
        raise ValueError("flanking_size must be a non-negative integer.")

    effective_max_peptide_length = (
        max_peptide_length + 2 * flanking_size
        if use_protein_features
        else max_peptide_length
    )
    required_max_length = resolve_max_length(tokenizer, effective_max_peptide_length)
    if max_length is None:
        max_length = required_max_length
    elif max_length < required_max_length:
        context_desc = (
            f"with flanking_size={flanking_size}"
            if use_protein_features
            else "without flanking"
        )
        raise ValueError(
            f"max_length={max_length} is too small for peptides up to "
            f"{effective_max_peptide_length} residues ({context_desc}) "
            f"and {required_max_length - effective_max_peptide_length} "
            "tokenizer special tokens."
        )
    dataframe = prepare_dataset_frames(
        dataset_name=dataset_name,
        data_dir=data_dir,
        max_peptide_length=max_peptide_length,
        use_protein_features=use_protein_features,
        flanking_size=flanking_size,
    )
    sequence_column = "flanked_seq" if use_protein_features else "peptide_seq"
    encodings = preprocess_for_protein_model(
        dataframe,
        tokenizer,
        model_type=model_type,
        max_length=max_length,
        sequence_column=sequence_column,
    )
    return PeptideDataset(
        encodings,
        group_ids=dataframe["parent_protein_id"].to_numpy(),
        descriptor_features=dataframe[list(DESCRIPTOR_COLUMNS)].to_numpy(
            dtype=np.float32
        ),
    )


def prepare_dataset_frames(
    dataset_name,
    data_dir="dataset",
    max_peptide_length=20,
    use_protein_features=False,
    flanking_size=5,
):
    """Load and clean labeled examples with descriptors and optional protein context."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_FILENAMES:
        raise ValueError(
            "dataset_name must be 'bcell' or 'sars'; input_covid.csv has no labels."
        )
    if max_peptide_length < 1:
        raise ValueError("max_peptide_length must be positive.")
    if flanking_size < 0:
        raise ValueError("flanking_size must be a non-negative integer.")

    csv_path = Path(data_dir) / DATASET_FILENAMES[dataset_name]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    dataframe = pd.read_csv(csv_path)
    required_columns = {
        "peptide_seq",
        "target",
        "parent_protein_id",
        *DESCRIPTOR_COLUMNS,
    }
    if use_protein_features:
        required_columns.update({"protein_seq", "start_position", "end_position"})
    missing = required_columns.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError(f"{csv_path.name} contains no rows.")

    dataframe = dataframe.copy().reset_index(drop=True)
    dataframe["peptide_seq"] = [
        _normalize_sequence(value, row_label=index)
        for index, value in dataframe["peptide_seq"].items()
    ]
    if dataframe["parent_protein_id"].isna().any():
        raise ValueError("'parent_protein_id' must not contain missing values.")

    if use_protein_features:
        if dataframe["protein_seq"].isna().any():
            raise ValueError("'protein_seq' must not contain missing values.")
        dataframe["protein_seq"] = [
            _normalize_sequence(value, row_label=index)
            for index, value in dataframe["protein_seq"].items()
        ]
        start_pos = pd.to_numeric(dataframe["start_position"], errors="coerce")
        end_pos = pd.to_numeric(dataframe["end_position"], errors="coerce")
        if start_pos.isna().any() or end_pos.isna().any():
            raise ValueError("'start_position' and 'end_position' must be valid numbers.")
        if not (start_pos % 1 == 0).all() or not (end_pos % 1 == 0).all():
            raise ValueError("'start_position' and 'end_position' must be integers.")
        if (start_pos < 1).any() or (end_pos < start_pos).any():
            raise ValueError(
                "Protein coordinates must be 1-based and end_position >= start_position."
            )
        dataframe["start_position"] = start_pos.astype("int64")
        dataframe["end_position"] = end_pos.astype("int64")

    labels = pd.to_numeric(dataframe["target"], errors="raise")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("'target' must contain only binary labels 0 and 1.")
    dataframe["target"] = labels.astype("int64")
    for column in DESCRIPTOR_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    input_rows = len(dataframe)
    overlength = dataframe["peptide_seq"].str.len().gt(max_peptide_length)
    removed_overlength = int(overlength.sum())
    dataframe = dataframe.loc[~overlength].copy()

    contradictory = dataframe.groupby("peptide_seq")["target"].transform(
        "nunique"
    ).gt(1)
    removed_conflicting = int(contradictory.sum())
    dataframe = dataframe.loc[~contradictory].copy()

    duplicates = dataframe["peptide_seq"].duplicated(keep="first")
    removed_duplicates = int(duplicates.sum())
    dataframe = dataframe.loc[~duplicates].reset_index(drop=True)
    if dataframe.empty or dataframe["target"].nunique() != 2:
        raise ValueError("The cleaned dataset must contain examples of both classes.")

    if use_protein_features:
        dataframe["flanked_seq"] = [
            extract_flanked_sequence(
                peptide_seq=row["peptide_seq"],
                protein_seq=row["protein_seq"],
                start_position=row["start_position"],
                end_position=row["end_position"],
                flanking_size=flanking_size,
                row_label=index,
            )
            for index, row in dataframe.iterrows()
        ]

    context_msg = (
        f"; flanking context: {flanking_size} residues (N- and C-terminal)"
        if use_protein_features
        else ""
    )
    print(
        f"TF cleaning ({dataset_name}): {input_rows:,} -> {len(dataframe):,} rows "
        f"(removed {removed_overlength:,} overlength, "
        f"{removed_conflicting:,} conflicting-label, "
        f"{removed_duplicates:,} duplicate rows){context_msg}."
    )
    print(
        "TF class counts: "
        f"{dataframe['target'].value_counts().sort_index().to_dict()}; "
        f"parent proteins: {dataframe['parent_protein_id'].nunique():,}."
    )
    return dataframe


def scale_descriptor_features(dataset, train_indices):
    """Fit imputation and scaling on training rows and transform all rows."""
    if dataset.descriptor_features is None:
        raise ValueError("Dataset does not contain descriptor features.")
    raw_features = dataset.descriptor_features.detach().cpu().numpy()
    train_features = raw_features[np.asarray(train_indices, dtype=np.int64)]
    if train_features.size == 0:
        raise ValueError("Training indices must select at least one example.")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_features)
    scaler = StandardScaler()
    scaler.fit(train_imputed)
    all_imputed = imputer.transform(raw_features)
    scaled_features = scaler.transform(all_imputed).astype(np.float32)
    dataset.descriptor_features = torch.tensor(
        scaled_features,
        dtype=torch.float32,
    )
    return imputer, scaler


def stratified_split_indices(
    labels,
    validation_size=0.2,
    random_state=42,
):
    """Split row indices into train and validation sets stratified by labels only."""
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("A stratified split requires non-empty binary labels.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    class_counts = np.bincount(labels.astype(int), minlength=2)
    if class_counts.min() < 2:
        raise ValueError("Each class needs at least two examples for a split.")

    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)),
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )
    if np.unique(labels[train_indices]).size != 2 or np.unique(
        labels[validation_indices]
    ).size != 2:
        raise ValueError("Both classes must appear in train and validation sets.")
    return train_indices, validation_indices
