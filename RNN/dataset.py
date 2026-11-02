from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


DATASET_FILENAMES = {
    "bcell": "input_bcell.csv",
    "sars": "input_sars.csv",
}
CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
RARE_AMINO_ACIDS = frozenset("BUZO")
AMINO_ACID_TO_ID = {
    residue: index + 1
    for index, residue in enumerate(CANONICAL_AMINO_ACIDS)
}
UNKNOWN_AMINO_ACID_ID = len(AMINO_ACID_TO_ID) + 1
PEPTIDE_START_ID = UNKNOWN_AMINO_ACID_ID + 1
PEPTIDE_END_ID = PEPTIDE_START_ID + 1
PAD_ID = 0
VOCAB_SIZE = PEPTIDE_END_ID + 1
BIOLOGICAL_FEATURE_COLUMNS = (
    "chou_fasman",
    "emini",
    "kolaskar_tongaonkar",
    "parker",
    "isoelectric_point",
    "aromaticity",
    "hydrophobicity",
    "stability",
)
TABULAR_FEATURE_COLUMNS = (
    "start_position",
    "end_position",
    *BIOLOGICAL_FEATURE_COLUMNS,
)


def normalize_sequence(value, row_label):
    if pd.isna(value):
        raise ValueError(f"Missing sequence at row {row_label}.")
    sequence = "".join(str(value).upper().split())
    if not sequence:
        raise ValueError(f"Empty sequence at row {row_label}.")
    allowed = set(CANONICAL_AMINO_ACIDS) | RARE_AMINO_ACIDS | {"X"}
    invalid = set(sequence).difference(allowed)
    if invalid:
        raise ValueError(
            f"Unsupported residues at row {row_label}: {sorted(invalid)}."
        )
    return "".join("X" if residue in RARE_AMINO_ACIDS else residue for residue in sequence)


def encode_sequence(sequence):
    """Map amino acids to integer IDs; zero is reserved for batch padding."""
    return [
        AMINO_ACID_TO_ID.get(residue, UNKNOWN_AMINO_ACID_ID)
        for residue in sequence
    ]


def encode_peptide_context(peptide_seq, context_seq, peptide_start):
    """Mark the target peptide span inside its protein-derived context."""
    peptide_end = peptide_start + len(peptide_seq)
    if context_seq[peptide_start:peptide_end] != peptide_seq:
        raise ValueError("Peptide sequence does not match its context span.")
    return (
        encode_sequence(context_seq[:peptide_start])
        + [PEPTIDE_START_ID]
        + encode_sequence(peptide_seq)
        + [PEPTIDE_END_ID]
        + encode_sequence(context_seq[peptide_end:])
    )


def extract_context_sequence(
    peptide_seq,
    protein_seq,
    start_position,
    end_position,
    flanking_size,
    row_label,
):
    """Use 1-based inclusive peptide coordinates to include protein flanks."""
    protein_length = len(protein_seq)
    start_index = start_position - 1
    end_index = end_position
    if (
        start_index < 0
        or end_index < start_index
        or end_index > protein_length
    ):
        raise ValueError(
            f"Invalid peptide coordinates [{start_position}, {end_position}] "
            f"for protein length {protein_length} at row {row_label}."
        )
    left = protein_seq[max(0, start_index - flanking_size):start_index]
    right = protein_seq[end_index:min(protein_length, end_index + flanking_size)]
    return f"{left}{peptide_seq}{right}"


def load_dataset(
    dataset_name="bcell",
    data_dir="dataset",
    max_peptide_length=20,
    flanking_size=5,
):
    """Load and clean rows with the same peptide-level rules as TF."""
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_FILENAMES:
        raise ValueError("dataset_name must be 'bcell' or 'sars'.")
    if max_peptide_length < 1:
        raise ValueError("max_peptide_length must be positive.")
    if flanking_size < 0:
        raise ValueError("flanking_size must be non-negative.")

    csv_path = Path(data_dir) / DATASET_FILENAMES[dataset_name]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    dataframe = pd.read_csv(csv_path)
    required = {
        "parent_protein_id",
        "protein_seq",
        "start_position",
        "end_position",
        "peptide_seq",
        "target",
        *BIOLOGICAL_FEATURE_COLUMNS,
    }
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}.")
    if dataframe.empty:
        raise ValueError(f"{csv_path.name} contains no rows.")

    dataframe = dataframe.copy().reset_index(drop=True)
    dataframe["peptide_seq"] = [
        normalize_sequence(value, index)
        for index, value in dataframe["peptide_seq"].items()
    ]
    dataframe["protein_seq"] = [
        normalize_sequence(value, index)
        for index, value in dataframe["protein_seq"].items()
    ]
    if dataframe["parent_protein_id"].isna().any():
        raise ValueError("'parent_protein_id' cannot contain missing values.")

    labels = pd.to_numeric(dataframe["target"], errors="raise")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("'target' must contain only binary values 0 and 1.")
    dataframe["target"] = labels.astype(np.int64)

    for column in ("start_position", "end_position"):
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric.isna().any() or not (numeric % 1 == 0).all():
            raise ValueError(f"'{column}' must contain valid integer coordinates.")
        dataframe[column] = numeric.astype(np.int64)
    if (
        (dataframe["start_position"] < 1).any()
        or (dataframe["end_position"] < dataframe["start_position"]).any()
    ):
        raise ValueError("Protein coordinates must be 1-based and end >= start.")

    for column in BIOLOGICAL_FEATURE_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    initial_count = len(dataframe)
    too_long = dataframe["peptide_seq"].str.len().gt(max_peptide_length)
    removed_long = int(too_long.sum())
    dataframe = dataframe.loc[~too_long].copy()
    conflicts = dataframe.groupby("peptide_seq")["target"].transform(
        "nunique"
    ).gt(1)
    removed_conflicts = int(conflicts.sum())
    dataframe = dataframe.loc[~conflicts].copy()
    duplicate = dataframe["peptide_seq"].duplicated(keep="first")
    removed_duplicates = int(duplicate.sum())
    dataframe = dataframe.loc[~duplicate].copy().reset_index(drop=True)
    if dataframe.empty or dataframe["target"].nunique() != 2:
        raise ValueError("The cleaned dataset must contain both label classes.")

    dataframe["context_seq"] = [
        extract_context_sequence(
            peptide_seq=row.peptide_seq,
            protein_seq=row.protein_seq,
            start_position=row.start_position,
            end_position=row.end_position,
            flanking_size=flanking_size,
            row_label=index,
        )
        for index, row in dataframe.iterrows()
    ]
    dataframe["peptide_start_in_context"] = [
        min(flanking_size, int(start_position) - 1)
        for start_position in dataframe["start_position"]
    ]
    maximum_context_length = max_peptide_length + 2 * flanking_size
    if dataframe["context_seq"].str.len().gt(maximum_context_length).any():
        raise ValueError("Flanked peptide exceeds the configured context length.")

    print(
        f"RNN cleaning ({dataset_name}): {initial_count:,} -> {len(dataframe):,} "
        f"(removed {removed_long:,} overlength, {removed_conflicts:,} "
        f"conflicting-label, {removed_duplicates:,} duplicate rows); "
        f"flank={flanking_size}."
    )
    print(
        "RNN class counts: "
        f"{dataframe['target'].value_counts().sort_index().to_dict()}."
    )
    return dataframe


class PeptideSequenceDataset(Dataset):
    """BiLSTM inputs: indexed flanked sequence, coordinates/descriptors, label."""

    def __init__(self, dataframe):
        self.sequences = [
            torch.tensor(
                encode_peptide_context(
                    peptide_seq=row.peptide_seq,
                    context_seq=row.context_seq,
                    peptide_start=row.peptide_start_in_context,
                ),
                dtype=torch.long,
            )
            for row in dataframe.itertuples(index=False)
        ]
        self.features = torch.tensor(
            dataframe[list(TABULAR_FEATURE_COLUMNS)].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            dataframe["target"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "sequence": self.sequences[index],
            "features": self.features[index],
            "labels": self.labels[index],
        }


def scale_tabular_features(dataset, train_indices):
    """Fit imputation and scaling on training rows only."""
    indices = np.asarray(train_indices, dtype=np.int64)
    raw_features = dataset.features.detach().cpu().numpy()
    if indices.size == 0:
        raise ValueError("Training indices must not be empty.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(raw_features[indices])
    scaler = StandardScaler().fit(train_imputed)
    dataset.features = torch.tensor(
        scaler.transform(imputer.transform(raw_features)).astype(np.float32),
        dtype=torch.float32,
    )
    return imputer, scaler


def collate_sequences(batch):
    """Pad variable-length sequences to the longest sequence in each batch."""
    lengths = torch.tensor(
        [sample["sequence"].numel() for sample in batch],
        dtype=torch.long,
    )
    max_length = int(lengths.max().item())
    sequences = torch.full((len(batch), max_length), PAD_ID, dtype=torch.long)
    for index, sample in enumerate(batch):
        sequence = sample["sequence"]
        sequences[index, : sequence.numel()] = sequence
    return {
        "sequence": sequences,
        "lengths": lengths,
        "features": torch.stack([sample["features"] for sample in batch]),
        "labels": torch.stack([sample["labels"] for sample in batch]),
    }