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
PERBANDINGAN_3_DESCRIPTOR_COLUMNS = (
    "hydrophobicity",
    "isoelectric_point",
    "emini",
    "parker",
)
BERT_CLS_PREFIX = "bert_cls_"
PROTTRANS_PREFIX = "prott5_"


def get_bert_cls_columns(columns) -> tuple[str, ...]:
    """Return an ordered CLS embedding schema from a feature table."""
    indexed_columns = []
    for column in columns:
        if not isinstance(column, str) or not column.startswith(BERT_CLS_PREFIX):
            continue
        suffix = column[len(BERT_CLS_PREFIX) :]
        if not suffix.isdigit():
            raise ValueError(f"Invalid BERT CLS feature column: {column!r}.")
        indexed_columns.append((int(suffix), column))
    indexed_columns.sort()
    indices = [index for index, _ in indexed_columns]
    if indices and indices != list(range(len(indices))):
        raise ValueError("BERT CLS feature indices must be contiguous from zero.")
    return tuple(column for _, column in indexed_columns)


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
    use_peptide_features: bool = True,
    use_protein_features: bool = False,
    use_biological_features: bool = True,
    peptide_feature_type: str = "aac",
    sequence_column: str = "peptide_seq",
    feature_set: str = "proposed",
    use_aac_features: bool | None = None,
    use_dpc_features: bool | None = None,
    parent_protein_id_mapping: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build one of the paper-based comparison profiles or proposed features."""
    if not isinstance(feature_set, str):
        raise ValueError(
            "feature_set must be 'perbandingan_1', 'perbandingan_2', "
            "'perbandingan_3', 'perbandingan_4', 'perbandingan_5', "
            "or 'proposed'."
        )
    feature_set = feature_set.strip().lower().replace(" ", "_")
    if feature_set not in {
        "perbandingan_1",
        "perbandingan_2",
        "perbandingan_3",
        "perbandingan_4",
        "perbandingan_5",
        "proposed",
    }:
        raise ValueError(
            "feature_set must be 'perbandingan_1', 'perbandingan_2', "
            "'perbandingan_3', 'perbandingan_4', 'perbandingan_5', "
            "or 'proposed'."
        )

    if feature_set == "perbandingan_1":
        required_columns = {
            sequence_column,
            "protein_seq",
            "start_position",
            "end_position",
            *DESCRIPTOR_COLUMNS,
        }
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                "The perbandingan_1 feature set requires columns: "
                f"{sorted(missing_columns)}."
            )
        peptide_sequences = [
            normalize_sequence(value, row_label=index)
            for index, value in dataframe[sequence_column].items()
        ]
        protein_sequences = [
            normalize_sequence(value, row_label=index)
            for index, value in dataframe["protein_seq"].items()
        ]
        start_positions = pd.to_numeric(dataframe["start_position"], errors="raise")
        end_positions = pd.to_numeric(dataframe["end_position"], errors="raise")
        if start_positions.isna().any() or end_positions.isna().any():
            raise ValueError("Protein start/end positions must not be missing.")
        if (end_positions < start_positions).any():
            raise ValueError("Protein end_position must be >= start_position.")

        features = pd.DataFrame(
            {
                "peptide_length": [len(sequence) for sequence in peptide_sequences],
                "protein_length": [len(sequence) for sequence in protein_sequences],
                "position": (end_positions - start_positions + 1).astype(float),
            },
            index=dataframe.index,
        )
        for column in DESCRIPTOR_COLUMNS:
            features[column] = pd.to_numeric(dataframe[column], errors="coerce")
        return features

    if feature_set == "perbandingan_2":
        required_columns = {
            sequence_column,
            "start_position",
            "end_position",
            *DESCRIPTOR_COLUMNS,
        }
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                "The perbandingan_2 feature set requires columns: "
                f"{sorted(missing_columns)}."
            )
        # Validate peptide sequences even though peptide_length follows the
        # paper's position formula rather than the sequence string length.
        for index, value in dataframe[sequence_column].items():
            normalize_sequence(value, row_label=index)
        start_positions = pd.to_numeric(
            dataframe["start_position"], errors="raise"
        )
        end_positions = pd.to_numeric(dataframe["end_position"], errors="raise")
        if start_positions.isna().any() or end_positions.isna().any():
            raise ValueError("Protein start/end positions must not be missing.")
        if not (start_positions % 1 == 0).all() or not (
            end_positions % 1 == 0
        ).all():
            raise ValueError("Protein start/end positions must be integers.")
        if (start_positions < 1).any() or (end_positions < start_positions).any():
            raise ValueError(
                "Protein positions must be 1-based and end_position must be "
                "greater than or equal to start_position."
            )

        features = pd.DataFrame(
            {
                "peptide_length": (
                    end_positions - start_positions + 1
                ).astype(float),
            },
            index=dataframe.index,
        )
        for column in DESCRIPTOR_COLUMNS:
            features[column] = pd.to_numeric(dataframe[column], errors="coerce")
        return features

    if feature_set == "perbandingan_3":
        required_columns = {
            sequence_column,
            "parent_protein_id",
            "protein_seq",
            "start_position",
            "end_position",
            *PERBANDINGAN_3_DESCRIPTOR_COLUMNS,
        }
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                "The perbandingan_3 feature set requires columns: "
                f"{sorted(missing_columns)}."
            )
        peptide_sequences = [
            normalize_sequence(value, row_label=index)
            for index, value in dataframe[sequence_column].items()
        ]
        protein_sequences = [
            normalize_sequence(value, row_label=index)
            for index, value in dataframe["protein_seq"].items()
        ]
        protein_ids = dataframe["parent_protein_id"]
        if protein_ids.isna().any():
            raise ValueError("parent_protein_id must not contain missing values.")
        protein_ids = protein_ids.astype(str)
        if parent_protein_id_mapping is None:
            categories = sorted(protein_ids.unique())
            parent_protein_id_mapping = {
                protein_id: category_index
                for category_index, protein_id in enumerate(categories)
            }
        if any(not isinstance(value, int) for value in parent_protein_id_mapping.values()):
            raise ValueError("parent_protein_id_mapping values must be integers.")

        start_positions = pd.to_numeric(dataframe["start_position"], errors="raise")
        end_positions = pd.to_numeric(dataframe["end_position"], errors="raise")
        if start_positions.isna().any() or end_positions.isna().any():
            raise ValueError("Protein start/end positions must not be missing.")
        if not (start_positions % 1 == 0).all() or not (
            end_positions % 1 == 0
        ).all():
            raise ValueError("Protein start/end positions must be integers.")
        if (start_positions < 1).any() or (end_positions < start_positions).any():
            raise ValueError(
                "Protein positions must be 1-based and end_position must be "
                "greater than or equal to start_position."
            )

        features = pd.DataFrame(
            {
                "parent_protein_id": [
                    parent_protein_id_mapping.get(protein_id, -1)
                    for protein_id in protein_ids
                ],
                "peptide_seq_length": [len(sequence) for sequence in peptide_sequences],
                "hydrophobicity": pd.to_numeric(
                    dataframe["hydrophobicity"], errors="coerce"
                ),
                "isoelectric_point": pd.to_numeric(
                    dataframe["isoelectric_point"], errors="coerce"
                ),
                "emini": pd.to_numeric(dataframe["emini"], errors="coerce"),
                "parker": pd.to_numeric(dataframe["parker"], errors="coerce"),
                "end_position": end_positions.astype(float),
                "start_position": start_positions.astype(float),
                "protein_seq_length": [len(sequence) for sequence in protein_sequences],
            },
            index=dataframe.index,
        )
        return features

    if feature_set == "perbandingan_4":
        bert_columns = get_bert_cls_columns(dataframe.columns)
        if sequence_column not in dataframe.columns:
            raise ValueError(
                f"The perbandingan_4 feature set requires {sequence_column!r}."
            )
        if not bert_columns:
            raise ValueError(
                "The perbandingan_4 feature set requires precomputed BERT CLS "
                "columns named bert_cls_000, bert_cls_001, and so on."
            )
        features = dataframe.loc[:, [sequence_column, *bert_columns]].copy()
        features[sequence_column] = [
            normalize_sequence(value, row_label=index)
            for index, value in features[sequence_column].items()
        ]
        for column in bert_columns:
            features[column] = pd.to_numeric(features[column], errors="raise")
        return features

    if feature_set == "perbandingan_5":
        embedding_columns = [
            column
            for column in dataframe.columns
            if isinstance(column, str) and column.startswith(PROTTRANS_PREFIX)
        ]
        if len(embedding_columns) != 1024 or embedding_columns != [
            f"prott5_{index:04d}" for index in range(1024)
        ]:
            raise ValueError(
                "The perbandingan_5 feature set requires 1024 ordered "
                "ProtT5 mean-embedding columns named prott5_0000...prott5_1023."
            )
        features = dataframe.loc[:, embedding_columns].copy()
        for column in embedding_columns:
            features[column] = pd.to_numeric(features[column], errors="raise")
        return features

    for name, enabled in (
        ("use_peptide_features", use_peptide_features),
        ("use_protein_features", use_protein_features),
        ("use_biological_features", use_biological_features),
    ):
        if not isinstance(enabled, bool):
            raise ValueError(f"{name} must be a boolean.")
    if not isinstance(peptide_feature_type, str):
        raise ValueError("peptide_feature_type must be a string.")
    peptide_feature_type = peptide_feature_type.lower()
    if peptide_feature_type not in {"aac", "aac_dipeptide"}:
        raise ValueError(
            "peptide_feature_type must be 'aac' or 'aac_dipeptide'."
        )
    use_aac_features = (
        use_peptide_features if use_aac_features is None else use_aac_features
    )
    use_dpc_features = (
        use_peptide_features and peptide_feature_type == "aac_dipeptide"
        if use_dpc_features is None
        else use_dpc_features
    )
    for name, enabled in (
        ("use_aac_features", use_aac_features),
        ("use_dpc_features", use_dpc_features),
    ):
        if not isinstance(enabled, bool):
            raise ValueError(f"{name} must be a boolean or None.")
    if not any(
        (
            use_aac_features,
            use_dpc_features,
            use_protein_features,
            use_biological_features,
        )
    ):
        raise ValueError("At least one feature group must be enabled.")
    if (use_aac_features or use_dpc_features or use_protein_features) and sequence_column not in dataframe:
        raise ValueError(f"DataFrame is missing sequence column {sequence_column!r}.")

    sequences = (
        [
            normalize_sequence(value, row_label=index)
            for index, value in dataframe[sequence_column].items()
        ]
        if use_aac_features or use_dpc_features or use_protein_features
        else []
    )
    features: dict[str, list[float]] = {}
    if use_aac_features:
        features.update(
            {f"aac_{amino_acid}": [] for amino_acid in AMINO_ACIDS}
        )
        features["aac_X"] = []
        features["sequence_length"] = []

    dipeptides = [first + second for first in AMINO_ACIDS for second in AMINO_ACIDS]
    if use_dpc_features:
        features.update({f"dpc_{dipeptide}": [] for dipeptide in dipeptides})

    protein_sequences = None
    if use_protein_features:
        required_columns = {"protein_seq", "start_position", "end_position"}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                "Protein features require columns: "
                f"{sorted(missing_columns)}."
            )
        protein_sequences = [
            normalize_sequence(value, row_label=index)
            for index, value in dataframe["protein_seq"].items()
        ]
        start_positions = pd.to_numeric(
            dataframe["start_position"],
            errors="raise",
        )
        end_positions = pd.to_numeric(
            dataframe["end_position"],
            errors="raise",
        )
        if start_positions.isna().any() or end_positions.isna().any():
            raise ValueError("Protein start/end positions must not be missing.")
        if not (start_positions % 1 == 0).all() or not (
            end_positions % 1 == 0
        ).all():
            raise ValueError("Protein start/end positions must be integers.")
        if (start_positions < 1).any() or (end_positions < start_positions).any():
            raise ValueError(
                "Protein positions must be 1-based and end_position must be "
                "greater than or equal to start_position."
            )
        for amino_acid in AMINO_ACIDS:
            features[f"protein_aac_{amino_acid}"] = []
        features["protein_aac_X"] = []
        features["start_position"] = []
        features["end_position"] = []

    if use_aac_features or use_dpc_features:
        for sequence in sequences:
            residue_counts = Counter(sequence)
            sequence_length = len(sequence)
            if use_aac_features:
                for amino_acid in AMINO_ACIDS:
                    features[f"aac_{amino_acid}"].append(
                        residue_counts[amino_acid] / sequence_length
                    )
                features["aac_X"].append(residue_counts["X"] / sequence_length)
                features["sequence_length"].append(float(sequence_length))

            if use_dpc_features:
                pair_counts = Counter(
                    sequence[index : index + 2]
                    for index in range(sequence_length - 1)
                )
                denominator = max(sequence_length - 1, 1)
                for dipeptide in dipeptides:
                    features[f"dpc_{dipeptide}"].append(
                        pair_counts[dipeptide] / denominator
                    )

    if use_protein_features:
        for index, protein_sequence in enumerate(protein_sequences):
            residue_counts = Counter(protein_sequence)
            protein_length = len(protein_sequence)
            for amino_acid in AMINO_ACIDS:
                features[f"protein_aac_{amino_acid}"].append(
                    residue_counts[amino_acid] / protein_length
                )
            features["protein_aac_X"].append(
                residue_counts["X"] / protein_length
            )
            features["start_position"].append(
                float(start_positions.iloc[index])
            )
            features["end_position"].append(float(end_positions.iloc[index]))

    result = pd.DataFrame(features, index=dataframe.index)
    if use_biological_features:
        missing_descriptors = set(DESCRIPTOR_COLUMNS).difference(
            dataframe.columns
        )
        if missing_descriptors:
            raise ValueError(
                "Biological features require descriptor columns: "
                f"{sorted(missing_descriptors)}."
            )
        for column in DESCRIPTOR_COLUMNS:
            result[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return result
