"""Load a saved ML bundle and predict labels for peptide sequences."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from ML.pre_dataset import (
    DESCRIPTOR_COLUMNS,
    PERBANDINGAN_3_DESCRIPTOR_COLUMNS,
    build_feature_frame,
    get_bert_cls_columns,
    normalize_sequence,
)


PROTTRANS_DIMENSION = 1024


def predict_peptides(
    sequences: list[str],
    model_path: str | Path = "ML/artifacts/peptide_classifier.joblib",
    descriptor_data: pd.DataFrame | None = None,
    protein_context_data: pd.DataFrame | None = None,
    bert_embedding_data: pd.DataFrame | None = None,
    prottrans_embedding_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict with precomputed model inputs, including ProtT5 mean vectors."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if not sequences:
        raise ValueError("Provide at least one peptide sequence.")

    bundle = load(model_path)
    feature_names = bundle["feature_names"]
    feature_profile = bundle.get("feature_profile", "proposed")
    legacy_feature_set = bundle.get("feature_set")
    use_protein_features = bool(
        bundle.get(
            "use_protein_features",
            any(name.startswith("protein_aac_") for name in feature_names)
            or {"start_position", "end_position"}.issubset(feature_names),
        )
    )
    use_peptide_features = bool(
        bundle.get(
            "use_peptide_features",
            legacy_feature_set in {"aac", "aac_dipeptide"}
            or use_protein_features
            or any(name.startswith("aac_") for name in feature_names),
        )
    )
    use_biological_features = bool(
        bundle.get(
            "use_biological_features",
            legacy_feature_set == "biological"
            or any(name in DESCRIPTOR_COLUMNS for name in feature_names),
        )
    )
    if feature_profile in {
        "perbandingan_1",
        "perbandingan_2",
        "perbandingan_3",
    }:
        use_peptide_features = False
        use_protein_features = False
        use_biological_features = False
    peptide_feature_type = bundle.get(
        "peptide_feature_type",
        legacy_feature_set
        if legacy_feature_set in {"aac", "aac_dipeptide"}
        else "aac",
    )
    use_aac_features = bool(
        bundle.get("use_aac_features", use_peptide_features)
    )
    use_dpc_features = bool(
        bundle.get(
            "use_dpc_features",
            use_peptide_features and peptide_feature_type == "aac_dipeptide",
        )
    )
    if feature_profile in {
        "perbandingan_1",
        "perbandingan_2",
        "perbandingan_3",
    }:
        use_aac_features = False
        use_dpc_features = False
    input_frame = pd.DataFrame({"peptide_seq": sequences})
    if feature_profile == "perbandingan_4":
        input_frame["peptide_seq"] = [
            normalize_sequence(value, row_label=index)
            for index, value in input_frame["peptide_seq"].items()
        ]
        if bert_embedding_data is None:
            raise ValueError(
                "A perbandingan_4 model needs bert_embedding_data with one row "
                "per sequence, keyed by peptide_seq and containing the saved "
                "bert_cls_000... columns. Supply embeddings from the same "
                "protein BERT encoder used for training."
            )
        if "peptide_seq" not in bert_embedding_data.columns:
            raise ValueError("bert_embedding_data must contain peptide_seq.")
        bert_columns = get_bert_cls_columns(bert_embedding_data.columns)
        expected_columns = tuple(
            name for name in feature_names if name.startswith("bert_cls_")
        )
        if not bert_columns or bert_columns != expected_columns:
            raise ValueError(
                "bert_embedding_data CLS columns must match the saved model "
                "schema exactly and be ordered bert_cls_000..."
            )
        embedding_frame = bert_embedding_data.loc[
            :, ["peptide_seq", *bert_columns]
        ].copy()
        embedding_frame["peptide_seq"] = [
            normalize_sequence(value, row_label=index)
            for index, value in embedding_frame["peptide_seq"].items()
        ]
        if embedding_frame["peptide_seq"].duplicated().any():
            raise ValueError("bert_embedding_data must have unique peptide_seq keys.")
        for column in bert_columns:
            embedding_frame[column] = pd.to_numeric(
                embedding_frame[column], errors="raise"
            )
        if not np.isfinite(embedding_frame.loc[:, bert_columns].to_numpy(dtype=float)).all():
            raise ValueError("bert_embedding_data contains non-finite CLS values.")
        input_frame = input_frame.assign(
            _input_order=range(len(input_frame))
        ).merge(
            embedding_frame,
            on="peptide_seq",
            how="left",
            validate="many_to_one",
            sort=False,
        ).sort_values("_input_order", kind="stable")
        if input_frame.loc[:, bert_columns].isna().any(axis=None):
            raise ValueError("bert_embedding_data is missing a requested peptide.")
        input_frame = input_frame.drop(columns="_input_order").reset_index(drop=True)
    elif feature_profile == "perbandingan_5":
        input_frame["peptide_seq"] = [
            normalize_sequence(value, row_label=index)
            for index, value in input_frame["peptide_seq"].items()
        ]
        if prottrans_embedding_data is None:
            raise ValueError(
                "A perbandingan_5 model needs prottrans_embedding_data with "
                "one row per unique peptide, keyed by peptide_seq and containing "
                "prott5_0000...prott5_1023 numeric columns."
            )
        if "peptide_seq" not in prottrans_embedding_data.columns:
            raise ValueError("prottrans_embedding_data must contain peptide_seq.")
        embedding_columns = [f"prott5_{index:04d}" for index in range(PROTTRANS_DIMENSION)]
        if tuple(feature_names) != tuple(embedding_columns):
            raise ValueError(
                "The saved perbandingan_5 model must use the ordered 1024-column "
                "ProtT5 schema prott5_0000...prott5_1023."
            )
        missing_columns = set(embedding_columns).difference(
            prottrans_embedding_data.columns
        )
        if missing_columns:
            raise ValueError(
                "prottrans_embedding_data is missing ProtT5 columns: "
                f"{sorted(missing_columns)[:5]}"
                f"{'...' if len(missing_columns) > 5 else ''}."
            )
        embedding_frame = prottrans_embedding_data.loc[
            :, ["peptide_seq", *embedding_columns]
        ].copy()
        embedding_frame["peptide_seq"] = [
            normalize_sequence(value, row_label=index)
            for index, value in embedding_frame["peptide_seq"].items()
        ]
        if embedding_frame["peptide_seq"].duplicated().any():
            raise ValueError(
                "prottrans_embedding_data must have unique peptide_seq keys."
            )
        for column in embedding_columns:
            embedding_frame[column] = pd.to_numeric(
                embedding_frame[column], errors="raise"
            )
        if not np.isfinite(
            embedding_frame.loc[:, embedding_columns].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "prottrans_embedding_data contains non-finite feature values."
            )
        input_frame = input_frame.assign(
            _input_order=range(len(input_frame))
        ).merge(
            embedding_frame,
            on="peptide_seq",
            how="left",
            validate="many_to_one",
            sort=False,
        ).sort_values("_input_order", kind="stable")
        if input_frame.loc[:, embedding_columns].isna().any(axis=None):
            raise ValueError(
                "prottrans_embedding_data is missing a requested peptide."
            )
        input_frame = input_frame.drop(columns="_input_order").reset_index(drop=True)
    if use_biological_features:
        if descriptor_data is None:
            raise ValueError(
                "This model uses precomputed physicochemical descriptors. "
                f"Pass descriptor_data containing: {list(DESCRIPTOR_COLUMNS)}."
            )
        missing = set(DESCRIPTOR_COLUMNS).difference(descriptor_data.columns)
        if missing:
            raise ValueError(
                f"descriptor_data is missing required columns: {sorted(missing)}."
            )
        if len(descriptor_data) != len(input_frame):
            raise ValueError("descriptor_data must have one row per peptide sequence.")
        for column in DESCRIPTOR_COLUMNS:
            input_frame[column] = descriptor_data[column].to_numpy()

    if feature_profile == "perbandingan_1":
        required_columns = {"protein_seq", "start_position", "end_position"}
        if protein_context_data is None:
            raise ValueError(
                "A perbandingan_1 model needs protein_context_data containing "
                "protein_seq, start_position, and end_position."
            )
        missing = required_columns.difference(protein_context_data.columns)
        if missing:
            raise ValueError(
                "protein_context_data is missing required columns: "
                f"{sorted(missing)}."
            )
        if len(protein_context_data) != len(input_frame):
            raise ValueError(
                "protein_context_data must have one row per peptide sequence."
            )
        for column in required_columns:
            input_frame[column] = protein_context_data[column].to_numpy()
        required_descriptors = set(feature_names).intersection(DESCRIPTOR_COLUMNS)
        if required_descriptors:
            if descriptor_data is None:
                raise ValueError(
                    "A perbandingan_1 model needs descriptor_data containing: "
                    f"{sorted(required_descriptors)}."
                )
            missing_descriptors = required_descriptors.difference(
                descriptor_data.columns
            )
            if missing_descriptors:
                raise ValueError(
                    "descriptor_data is missing required columns: "
                    f"{sorted(missing_descriptors)}."
                )
            if len(descriptor_data) != len(input_frame):
                raise ValueError(
                    "descriptor_data must have one row per peptide sequence."
                )
            for column in required_descriptors:
                input_frame[column] = descriptor_data[column].to_numpy()
    elif feature_profile == "perbandingan_2":
        required_columns = {"start_position", "end_position"}
        if protein_context_data is None:
            raise ValueError(
                "A perbandingan_2 model needs protein_context_data containing "
                "start_position and end_position."
            )
        missing = required_columns.difference(protein_context_data.columns)
        if missing:
            raise ValueError(
                "protein_context_data is missing required columns: "
                f"{sorted(missing)}."
            )
        if len(protein_context_data) != len(input_frame):
            raise ValueError(
                "protein_context_data must have one row per peptide sequence."
            )
        if descriptor_data is None:
            raise ValueError(
                "A perbandingan_2 model needs descriptor_data containing: "
                f"{list(DESCRIPTOR_COLUMNS)}."
            )
        missing_descriptors = set(DESCRIPTOR_COLUMNS).difference(
            descriptor_data.columns
        )
        if missing_descriptors:
            raise ValueError(
                "descriptor_data is missing required columns: "
                f"{sorted(missing_descriptors)}."
            )
        if len(descriptor_data) != len(input_frame):
            raise ValueError(
                "descriptor_data must have one row per peptide sequence."
            )
        for column in required_columns:
            input_frame[column] = protein_context_data[column].to_numpy()
        for column in DESCRIPTOR_COLUMNS:
            input_frame[column] = descriptor_data[column].to_numpy()
    elif feature_profile == "perbandingan_3":
        required_columns = {
            "parent_protein_id",
            "protein_seq",
            "start_position",
            "end_position",
        }
        if protein_context_data is None:
            raise ValueError(
                "A perbandingan_3 model needs protein_context_data containing "
                f"{sorted(required_columns)}."
            )
        missing = required_columns.difference(protein_context_data.columns)
        if missing:
            raise ValueError(
                "protein_context_data is missing required columns: "
                f"{sorted(missing)}."
            )
        if len(protein_context_data) != len(input_frame):
            raise ValueError(
                "protein_context_data must have one row per peptide sequence."
            )
        if descriptor_data is None:
            raise ValueError(
                "A perbandingan_3 model needs descriptor_data containing: "
                f"{list(PERBANDINGAN_3_DESCRIPTOR_COLUMNS)}."
            )
        missing_descriptors = set(PERBANDINGAN_3_DESCRIPTOR_COLUMNS).difference(
            descriptor_data.columns
        )
        if missing_descriptors:
            raise ValueError(
                "descriptor_data is missing required columns: "
                f"{sorted(missing_descriptors)}."
            )
        if len(descriptor_data) != len(input_frame):
            raise ValueError(
                "descriptor_data must have one row per peptide sequence."
            )
        for column in required_columns:
            input_frame[column] = protein_context_data[column].to_numpy()
        for column in PERBANDINGAN_3_DESCRIPTOR_COLUMNS:
            input_frame[column] = descriptor_data[column].to_numpy()
    elif use_protein_features:
        required_columns = {"protein_seq", "start_position", "end_position"}
        if protein_context_data is None:
            raise ValueError(
                "This model uses protein-context features. Pass "
                "protein_context_data containing protein_seq, start_position, "
                "and end_position."
            )
        missing = required_columns.difference(protein_context_data.columns)
        if missing:
            raise ValueError(
                "protein_context_data is missing required columns: "
                f"{sorted(missing)}."
            )
        if len(protein_context_data) != len(input_frame):
            raise ValueError(
                "protein_context_data must have one row per peptide sequence."
            )
        for column in required_columns:
            input_frame[column] = protein_context_data[column].to_numpy()

    feature_frame = build_feature_frame(
        input_frame,
        use_peptide_features=use_peptide_features,
        use_protein_features=use_protein_features,
        use_biological_features=use_biological_features,
        peptide_feature_type=peptide_feature_type,
        feature_set=feature_profile,
        use_aac_features=use_aac_features,
        use_dpc_features=use_dpc_features,
        parent_protein_id_mapping=bundle.get("parent_protein_id_mapping"),
    ).reindex(columns=feature_names)
    probabilities = bundle["model"].predict_proba(feature_frame)[:, 1]
    threshold = float(bundle["prediction_threshold"])
    return pd.DataFrame(
        {
            "peptide_seq": input_frame["peptide_seq"],
            "probability_positive": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
        }
    )
