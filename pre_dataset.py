from pathlib import Path

import pandas as pd


def load_protbert_tokenizer(model_name="Rostlab/prot_bert"):
    """Load ProtBert's tokenizer using the BertTokenizer class."""
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained(model_name, do_lower_case=False)


def load_labeled_datasets(*csv_paths):
    """Load and combine labeled peptide CSV files with matching columns."""
    if not csv_paths:
        raise ValueError("Provide at least one CSV path.")

    datasets = [pd.read_csv(path) for path in csv_paths]
    required_columns = {"peptide_seq", "target"}
    for path, dataset in zip(csv_paths, datasets):
        missing = required_columns.difference(dataset.columns)
        if missing:
            raise ValueError(f"{Path(path).name} is missing columns: {sorted(missing)}")

    return pd.concat(datasets, ignore_index=True)


def preprocess_for_protbert(
    dataframe,
    tokenizer,
    max_length=32,
    sequence_column="peptide_seq",
    label_column="target",
):
    """Normalize peptide sequences and tokenize them for ProtBert classification."""
    if max_length < 1:
        raise ValueError("max_length must be a positive integer.")
    if sequence_column not in dataframe or label_column not in dataframe:
        raise ValueError(
            f"DataFrame must contain {sequence_column!r} and {label_column!r}."
        )
    if dataframe[[sequence_column, label_column]].isna().any().any():
        raise ValueError("Sequences and labels must not contain missing values.")

    canonical_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    rare_amino_acids = set("BUZO")
    sequences = []
    for index, value in dataframe[sequence_column].items():
        sequence = "".join(str(value).upper().split())
        if not sequence:
            raise ValueError(f"Empty peptide sequence at row {index}.")

        invalid = set(sequence).difference(canonical_amino_acids | rare_amino_acids | {"X"})
        if invalid:
            raise ValueError(
                f"Unsupported residues at row {index}: {sorted(invalid)}"
            )
        sequences.append(
            " ".join("X" if residue in rare_amino_acids else residue for residue in sequence)
        )

    labels = pd.to_numeric(dataframe[label_column], errors="raise")
    if not labels.isin((0, 1)).all():
        raise ValueError(f"{label_column!r} must contain only binary labels 0 and 1.")
    labels = labels.astype("int64")

    encoded = tokenizer(
        sequences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    encoded["labels"] = labels.tolist()
    return encoded
