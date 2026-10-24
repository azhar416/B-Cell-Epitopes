"""Sequence normalization and fixed-length feature extraction."""

from collections import Counter

import pandas as pd


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
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


def normalize_sequence(value: object, row_label: object = "unknown") -> str:
    """Normalize whitespace/case and map ambiguous amino acids to X."""
    if pd.isna(value):
        raise ValueError(f"Missing peptide sequence at row {row_label}.")
    sequence = "".join(str(value).upper().split())
    if not sequence:
        raise ValueError(f"Empty peptide sequence at row {row_label}.")

    invalid = set(sequence).difference(set(AMINO_ACIDS) | RARE_AMINO_ACIDS | {"X"})
    if invalid:
        raise ValueError(
            f"Unsupported residues at row {row_label}: {sorted(invalid)}."
        )
    return "".join("X" if residue in RARE_AMINO_ACIDS else residue for residue in sequence)


def build_feature_frame(
    dataframe: pd.DataFrame,
    feature_set: str = "aac",
    sequence_column: str = "peptide_seq",
) -> pd.DataFrame:
    """Return AAC, length, optional dipeptide, and available descriptor features."""
    feature_set = feature_set.lower()
    if feature_set not in {"aac", "aac_dipeptide"}:
        raise ValueError("feature_set must be 'aac' or 'aac_dipeptide'.")
    if sequence_column not in dataframe:
        raise ValueError(f"DataFrame is missing sequence column {sequence_column!r}.")

    sequences = [
        normalize_sequence(value, row_label=index)
        for index, value in dataframe[sequence_column].items()
    ]
    features: dict[str, list[float]] = {
        f"aac_{amino_acid}": [] for amino_acid in AMINO_ACIDS
    }
    features["aac_X"] = []
    features["sequence_length"] = []

    dipeptides = [first + second for first in AMINO_ACIDS for second in AMINO_ACIDS]
    if feature_set == "aac_dipeptide":
        features.update({f"dpc_{dipeptide}": [] for dipeptide in dipeptides})

    for sequence in sequences:
        residue_counts = Counter(sequence)
        sequence_length = len(sequence)
        for amino_acid in AMINO_ACIDS:
            features[f"aac_{amino_acid}"].append(
                residue_counts[amino_acid] / sequence_length
            )
        features["aac_X"].append(residue_counts["X"] / sequence_length)
        features["sequence_length"].append(float(sequence_length))

        if feature_set == "aac_dipeptide":
            pair_counts = Counter(
                sequence[index : index + 2]
                for index in range(sequence_length - 1)
            )
            denominator = max(sequence_length - 1, 1)
            for dipeptide in dipeptides:
                features[f"dpc_{dipeptide}"].append(
                    pair_counts[dipeptide] / denominator
                )

    result = pd.DataFrame(features, index=dataframe.index)
    for column in DESCRIPTOR_COLUMNS:
        if column in dataframe.columns:
            result[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return result