"""Load a saved ML bundle and predict labels for peptide sequences."""

from pathlib import Path

import pandas as pd
from joblib import load

from ML.pre_dataset import DESCRIPTOR_COLUMNS, build_feature_frame


def predict_peptides(
    sequences: list[str],
    model_path: str | Path = "ML/artifacts/peptide_classifier.joblib",
    descriptor_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predict sequences, requiring descriptors when the saved model uses them."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    if not sequences:
        raise ValueError("Provide at least one peptide sequence.")

    bundle = load(model_path)
    used_descriptors = [
        name for name in bundle["feature_names"] if name in DESCRIPTOR_COLUMNS
    ]
    input_frame = pd.DataFrame({"peptide_seq": sequences})
    if used_descriptors:
        if descriptor_data is None:
            raise ValueError(
                "This model uses precomputed physicochemical descriptors. "
                f"Pass descriptor_data containing: {used_descriptors}."
            )
        missing = set(used_descriptors).difference(descriptor_data.columns)
        if missing:
            raise ValueError(
                f"descriptor_data is missing required columns: {sorted(missing)}."
            )
        if len(descriptor_data) != len(input_frame):
            raise ValueError("descriptor_data must have one row per peptide sequence.")
        for column in used_descriptors:
            input_frame[column] = descriptor_data[column].to_numpy()

    feature_frame = build_feature_frame(
        input_frame,
        feature_set=bundle["feature_set"],
    ).reindex(columns=bundle["feature_names"])
    probabilities = bundle["model"].predict_proba(feature_frame)[:, 1]
    threshold = float(bundle["prediction_threshold"])
    return pd.DataFrame(
        {
            "peptide_seq": input_frame["peptide_seq"],
            "probability_positive": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
        }
    )
