"""Sequence normalization and fixed-length feature extraction."""

from collections import Counter

import pandas as pd


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
RARE_AMINO_ACIDS = frozenset("BUZO")
# Peptide-level propensity scales: these vary between peptides of one protein.
PROP_COLUMNS = (
    "chou_fasman",
    "emini",
    "kolaskar_tongaonkar",
    "parker",
)
# Computed from protein_seq, so every peptide of a protein carries the same
# value (measured intra-class correlation 1.000). Under protein-grouped folds
# they identify the protein instead of describing the peptide, which is why
# they form their own group rather than sitting inside the propensity block.
PROTGLOBAL_COLUMNS = (
    "isoelectric_point",
    "aromaticity",
    "hydrophobicity",
    "stability",
)
# The historical "biological" block, kept so earlier runs stay reproducible.
DESCRIPTOR_COLUMNS = PROP_COLUMNS + PROTGLOBAL_COLUMNS
AAC_COLUMNS = tuple(f"aac_{amino_acid}" for amino_acid in AMINO_ACIDS)
PERBANDINGAN_3_DESCRIPTOR_COLUMNS = (
    "hydrophobicity",
    "isoelectric_point",
    "emini",
    "parker",
)
BERT_CLS_PREFIX = "bert_cls_"
PROTTRANS_PREFIX = "prott5_"
# Residues taken from the parent protein on each side of the peptide. Fixed in
# advance rather than tuned: a B-cell linear epitope is typically 8-12 residues,
# and every value tested would widen the selection space the 1-SE rule guards.
FLANK_WINDOW_DEFAULT = 3


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
    use_flank_features: bool = False,
    flank_window: int = FLANK_WINDOW_DEFAULT,
    flank_split_sides: bool = True,
    use_len_features: bool = False,
    use_prop_features: bool = False,
    use_protglobal_features: bool = False,
    use_protlen_features: bool = False,
    use_pos_features: bool = False,
    use_protid_features: bool = False,
) -> pd.DataFrame:
    """Build one of the paper-based comparison profiles or proposed features."""
    if not isinstance(feature_set, str):
        raise ValueError(
            "feature_set must be 'perbandingan_1', 'perbandingan_2', "
            "'perbandingan_3', 'perbandingan_4', 'perbandingan_5', "
            "'kontras', or 'proposed'."
        )
    feature_set = feature_set.strip().lower().replace(" ", "_")
    if feature_set not in {
        "perbandingan_1",
        "perbandingan_2",
        "perbandingan_3",
        "perbandingan_4",
        "perbandingan_5",
        "kontras",
        "proposed",
    }:
        raise ValueError(
            "feature_set must be 'perbandingan_1', 'perbandingan_2', "
            "'perbandingan_3', 'perbandingan_4', 'perbandingan_5', "
            "'kontras', or 'proposed'."
        )

    if feature_set == "perbandingan_1":
        # Six columns, following the paper's own feature engineering. It drops
        # the protein id, start_position and end_position, replacing the last
        # two with "position" = (end - start) + 1. It also drops protein_length
        # ("it was not used in this study") and isoelectric_point, aromaticity,
        # hydrophobicity and stability, which the paper notes "are the same in
        # all peptides as they are properties dependent on the protein
        # sequence". Only the four peptide-level propensity scales remain, so
        # protein_seq is not needed at all.
        required_columns = {
            sequence_column,
            "start_position",
            "end_position",
            *PROP_COLUMNS,
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
        start_positions = pd.to_numeric(dataframe["start_position"], errors="raise")
        end_positions = pd.to_numeric(dataframe["end_position"], errors="raise")
        if start_positions.isna().any() or end_positions.isna().any():
            raise ValueError("Protein start/end positions must not be missing.")
        if (end_positions < start_positions).any():
            raise ValueError("Protein end_position must be >= start_position.")

        features = pd.DataFrame(
            {
                "peptide_length": [len(sequence) for sequence in peptide_sequences],
                # Duplicates peptide_length on this data, where the recorded
                # coordinates always span the peptide exactly. Kept because the
                # paper lists both.
                "position": (end_positions - start_positions + 1).astype(float),
            },
            index=dataframe.index,
        )
        for column in PROP_COLUMNS:
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
        ("use_flank_features", use_flank_features),
        ("flank_split_sides", flank_split_sides),
        ("use_len_features", use_len_features),
        ("use_prop_features", use_prop_features),
        ("use_protglobal_features", use_protglobal_features),
        ("use_protlen_features", use_protlen_features),
        ("use_pos_features", use_pos_features),
        ("use_protid_features", use_protid_features),
    ):
        if not isinstance(enabled, bool):
            raise ValueError(f"{name} must be a boolean.")
    # The historical block is exactly the two level-specific groups, in the
    # same column order, so earlier runs reproduce bit for bit.
    if use_biological_features:
        use_prop_features = True
        use_protglobal_features = True
    if not isinstance(flank_window, int) or isinstance(flank_window, bool):
        raise ValueError("flank_window must be an integer.")
    if flank_window < 1:
        raise ValueError("flank_window must be positive.")
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
            use_flank_features,
            use_len_features,
            use_prop_features,
            use_protglobal_features,
            use_protlen_features,
            use_pos_features,
            use_protid_features,
        )
    ):
        raise ValueError("At least one feature group must be enabled.")
    needs_sequence = (
        use_aac_features
        or use_dpc_features
        or use_protein_features
        or use_flank_features
        or use_len_features
    )
    if needs_sequence and sequence_column not in dataframe:
        raise ValueError(f"DataFrame is missing sequence column {sequence_column!r}.")

    sequences = (
        [
            normalize_sequence(value, row_label=index)
            for index, value in dataframe[sequence_column].items()
        ]
        if needs_sequence
        else []
    )
    features: dict[str, list[float]] = {}
    if use_aac_features:
        # 20 residue frequencies only. aac_X is absent because no ambiguous
        # residue survives cleaning in either dataset, and peptide length is
        # its own group rather than a passenger inside composition.
        features.update({column: [] for column in AAC_COLUMNS})
    if use_len_features:
        # Named as the reference profiles name it, so a reconstruction of
        # FS-1/FS-2 from our own groups is identical column for column.
        features["peptide_length"] = []

    dipeptides = [first + second for first in AMINO_ACIDS for second in AMINO_ACIDS]
    if use_dpc_features:
        features.update({f"dpc_{dipeptide}": [] for dipeptide in dipeptides})

    protein_sequences = None
    start_positions = None
    end_positions = None
    needs_protein_context = (
        use_protein_features
        or use_flank_features
        or use_protlen_features
        or use_pos_features
    )
    if needs_protein_context:
        required_columns = {"protein_seq", "start_position", "end_position"}
        missing_columns = required_columns.difference(dataframe.columns)
        if missing_columns:
            raise ValueError(
                "Protein and flank features require columns: "
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
    if use_protein_features:
        for amino_acid in AMINO_ACIDS:
            features[f"protein_aac_{amino_acid}"] = []
        features["protein_aac_X"] = []
        features["start_position"] = []
        features["end_position"] = []

    flank_sides = ("left", "right") if flank_split_sides else ("",)
    if use_flank_features:
        for side in flank_sides:
            prefix = f"flank_{side}_aac_" if side else "flank_aac_"
            for amino_acid in AMINO_ACIDS:
                features[f"{prefix}{amino_acid}"] = []
        # How many residues each side actually supplied. Without these a
        # truncated flank is indistinguishable from a balanced one, because
        # both are rates and an absent flank is an all-zero vector.
        features["flank_left_available"] = []
        features["flank_right_available"] = []

    if use_protlen_features:
        features["protein_length"] = []
    if use_pos_features:
        features["start_position"] = []
        features["end_position"] = []

    if use_aac_features or use_dpc_features or use_len_features:
        for sequence in sequences:
            residue_counts = Counter(sequence)
            sequence_length = len(sequence)
            if use_aac_features:
                for amino_acid in AMINO_ACIDS:
                    features[f"aac_{amino_acid}"].append(
                        residue_counts[amino_acid] / sequence_length
                    )
            if use_len_features:
                features["peptide_length"].append(float(sequence_length))

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

    if use_flank_features:
        for index, protein_sequence in enumerate(protein_sequences):
            start = int(start_positions.iloc[index])
            end = int(end_positions.iloc[index])
            # The flank is only meaningful if the coordinates really point at
            # the peptide. A silent mismatch would read neighbouring residues
            # from the wrong place and never raise.
            if protein_sequence[start - 1 : end] != sequences[index]:
                raise ValueError(
                    "Flank features need start_position/end_position to locate "
                    f"the peptide in protein_seq; row {dataframe.index[index]!r} "
                    f"has {sequences[index]!r} at positions {start}-{end} but "
                    f"protein_seq holds {protein_sequence[start - 1 : end]!r}."
                )
            left = protein_sequence[max(0, start - 1 - flank_window) : start - 1]
            right = protein_sequence[end : end + flank_window]
            for side in flank_sides:
                prefix = f"flank_{side}_aac_" if side else "flank_aac_"
                flank = {"left": left, "right": right, "": left + right}[side]
                residue_counts = Counter(flank)
                # Rate over the residues that exist, never padded with a
                # placeholder: padding would invent composition that is not
                # in the protein.
                denominator = max(len(flank), 1)
                for amino_acid in AMINO_ACIDS:
                    features[f"{prefix}{amino_acid}"].append(
                        residue_counts[amino_acid] / denominator
                    )
            features["flank_left_available"].append(float(len(left)))
            features["flank_right_available"].append(float(len(right)))

    if use_protlen_features:
        for protein_sequence in protein_sequences:
            features["protein_length"].append(float(len(protein_sequence)))
    if use_pos_features:
        for index in range(len(dataframe)):
            features["start_position"].append(float(start_positions.iloc[index]))
            features["end_position"].append(float(end_positions.iloc[index]))

    result = pd.DataFrame(features, index=dataframe.index)
    # PROP before PROTGLOBAL, so enabling both reproduces the historical
    # DESCRIPTOR_COLUMNS order exactly.
    for enabled, columns, label in (
        (use_prop_features, PROP_COLUMNS, "Propensity"),
        (use_protglobal_features, PROTGLOBAL_COLUMNS, "Protein-global"),
    ):
        if not enabled:
            continue
        missing_descriptors = set(columns).difference(dataframe.columns)
        if missing_descriptors:
            raise ValueError(
                f"{label} features require descriptor columns: "
                f"{sorted(missing_descriptors)}."
            )
        for column in columns:
            result[column] = pd.to_numeric(dataframe[column], errors="coerce")

    if use_protid_features:
        if "parent_protein_id" not in dataframe:
            raise ValueError(
                "protid features require a parent_protein_id column."
            )
        protein_ids = dataframe["parent_protein_id"].astype(str)
        mapping = parent_protein_id_mapping
        if mapping is None:
            mapping = {
                protein_id: index
                for index, protein_id in enumerate(sorted(protein_ids.unique()))
            }
        # Unseen proteins collapse to -1, exactly as the perbandingan_3
        # profile does. On the SARS set that is every single row.
        result["parent_protein_id"] = [
            mapping.get(protein_id, -1) for protein_id in protein_ids
        ]
    return result
