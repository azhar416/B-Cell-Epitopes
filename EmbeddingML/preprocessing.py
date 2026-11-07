"""Dataset validation and cleaning compatible with the ML and TF workflows."""

from pathlib import Path

import numpy as np
import pandas as pd


DATASET_FILENAMES = {
    "bcell": "input_bcell.csv",
    "sars": "input_sars.csv",
}
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
RARE_AMINO_ACIDS = frozenset("BUZO")
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


def normalize_sequence(value: object, row_label: object) -> str:
    """Normalize sequence text and map rare amino acids to X, as in ML/TF."""
    if pd.isna(value):
        raise ValueError(f"Missing sequence at row {row_label}.")
    sequence = "".join(str(value).upper().split())
    if not sequence:
        raise ValueError(f"Empty sequence at row {row_label}.")
    invalid = set(sequence).difference(CANONICAL_AMINO_ACIDS | RARE_AMINO_ACIDS | {"X"})
    if invalid:
        raise ValueError(
            f"Unsupported residues at row {row_label}: {sorted(invalid)}."
        )
    return "".join("X" if residue in RARE_AMINO_ACIDS else residue for residue in sequence)


def clean_dataframe(
    dataframe: pd.DataFrame,
    *,
    max_peptide_length: int = 20,
    source_name: str = "dataset",
) -> pd.DataFrame:
    """Validate schema and apply shared label, length, conflict, and dedup rules."""
    if max_peptide_length < 1:
        raise ValueError("max_peptide_length must be positive.")
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
        raise ValueError(f"{source_name} is missing columns: {sorted(missing)}.")
    if dataframe.empty:
        raise ValueError(f"{source_name} contains no rows.")

    cleaned = dataframe.copy().reset_index(drop=True)
    cleaned.insert(0, "source_row_index", np.arange(len(cleaned), dtype=np.int64))
    for column in ("peptide_seq", "protein_seq"):
        cleaned[column] = [
            normalize_sequence(value, row_label=index)
            for index, value in cleaned[column].items()
        ]
    if cleaned["parent_protein_id"].isna().any():
        raise ValueError("'parent_protein_id' must not contain missing values.")
    cleaned["parent_protein_id"] = cleaned["parent_protein_id"].astype(str)

    labels = pd.to_numeric(cleaned["target"], errors="raise")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("'target' must contain only binary values 0 and 1.")
    cleaned["target"] = labels.astype(np.int64)
    for column in ("start_position", "end_position"):
        values = pd.to_numeric(cleaned[column], errors="coerce")
        if values.isna().any() or not (values % 1 == 0).all():
            raise ValueError(f"'{column}' must contain valid integer coordinates.")
        cleaned[column] = values.astype(np.int64)
    if (
        (cleaned["start_position"] < 1).any()
        or (cleaned["end_position"] < cleaned["start_position"]).any()
    ):
        raise ValueError("Protein coordinates must be 1-based and end >= start.")
    for column in BIOLOGICAL_FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    initial_count = len(cleaned)
    overlength = cleaned["peptide_seq"].str.len().gt(max_peptide_length)
    removed_overlength = int(overlength.sum())
    cleaned = cleaned.loc[~overlength].copy()

    conflicting = cleaned.groupby("peptide_seq")["target"].transform(
        "nunique"
    ).gt(1)
    removed_conflicting = int(conflicting.sum())
    cleaned = cleaned.loc[~conflicting].copy()

    duplicates = cleaned["peptide_seq"].duplicated(keep="first")
    removed_duplicates = int(duplicates.sum())
    cleaned = cleaned.loc[~duplicates].reset_index(drop=True)
    if cleaned.empty or cleaned["target"].nunique() != 2:
        raise ValueError(
            "The cleaned dataset must contain examples from both label classes."
        )

    print(
        f"EmbeddingML cleaning ({source_name}): {initial_count:,} -> "
        f"{len(cleaned):,} rows (removed {removed_overlength:,} overlength, "
        f"{removed_conflicting:,} conflicting-label, "
        f"{removed_duplicates:,} duplicate rows)."
    )
    print(
        f"Class counts: {cleaned['target'].value_counts().sort_index().to_dict()}; "
        f"parent proteins: {cleaned['parent_protein_id'].nunique():,}."
    )
    return cleaned


def load_labeled_dataset(
    dataset_name: str = "bcell",
    data_dir: str | Path = "dataset",
    max_peptide_length: int = 20,
) -> pd.DataFrame:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_FILENAMES:
        raise ValueError("dataset_name must be 'bcell' or 'sars'.")
    csv_path = Path(data_dir) / DATASET_FILENAMES[dataset_name]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    return clean_dataframe(
        pd.read_csv(csv_path),
        max_peptide_length=max_peptide_length,
        source_name=dataset_name,
    )
