"""Load a saved ML bundle and predict labels for peptide sequences."""

from pathlib import Path

import pandas as pd
from joblib import load

from ML.pre_dataset import DESCRIPTOR_COLUMNS, build_feature_frame


def predict_peptides(
    sequences: list[str],
    model_path: str | Path = "ML/artifacts/peptide_classifier.joblib",
    descriptor_data: pd.DataFrame | None = None,
    protein_context_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict peptides, requiring optional descriptors and protein context."""
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
    if feature_profile == "perbandingan_1":
        use_peptide_features = False
        use_protein_features = False
        use_biological_features = False
    peptide_feature_type = bundle.get(
        "peptide_feature_type",
        legacy_feature_set
        if legacy_feature_set in {"aac", "aac_dipeptide"}
        else "aac",
    )
    input_frame = pd.DataFrame({"peptide_seq": sequences})
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
        required_columns = {"protein_seq"}
        if protein_context_data is None:
            raise ValueError(
                "A perbandingan_1 model needs protein_context_data containing "
                "protein_seq so it can calculate protein length."
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
        input_frame["protein_seq"] = protein_context_data["protein_seq"].to_numpy()
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
