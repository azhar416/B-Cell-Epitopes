from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    """Configuration for preprocessing, stratified validation, and training."""

    DATASET_NAME: str = "bcell"
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    EVALUATION_DATA_DIR: Path | None = None
    OUTPUT_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "ML" / "artifacts")
    MODEL_TYPE: str = "logistic_regression"
    TRAINING_MODE: str = "same_dataset"
    USE_PEPTIDE_FEATURES: bool = False
    USE_PROTEIN_FEATURES: bool = False
    USE_BIOLOGICAL_FEATURES: bool = True
    PEPTIDE_FEATURE_TYPE: str = "aac"
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    USE_SMOTE: bool = True
    SMOTE_K_NEIGHBORS: int = 5
    MAX_PEPTIDE_LENGTH: int = 20
    PREDICTION_THRESHOLD: float = 0.5
    RANDOM_FOREST_TREES: int = 300

    def validate(self) -> None:
        if self.DATASET_NAME.lower() not in {"bcell", "sars"}:
            raise ValueError("DATASET_NAME must be either 'bcell' or 'sars'.")
        if self.MODEL_TYPE.lower() not in {
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "svm",
        }:
            raise ValueError(
                "MODEL_TYPE must be logistic_regression, random_forest, "
                "extra_trees, or svm."
            )
        if not isinstance(self.TRAINING_MODE, str):
            raise ValueError("TRAINING_MODE must be a string.")
        self.TRAINING_MODE = self.TRAINING_MODE.lower()
        if self.TRAINING_MODE not in {"same_dataset", "bcell_to_sars"}:
            raise ValueError(
                "TRAINING_MODE must be 'same_dataset' or 'bcell_to_sars'."
            )
        if self.TRAINING_MODE == "bcell_to_sars" and self.VALIDATION_SIZE != 0.2:
            raise ValueError(
                "bcell_to_sars mode evaluates a fixed 20% SARS holdout; "
                "VALIDATION_SIZE must be 0.2."
            )
        for name in (
            "USE_PEPTIDE_FEATURES",
            "USE_PROTEIN_FEATURES",
            "USE_BIOLOGICAL_FEATURES",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if self.USE_PROTEIN_FEATURES and not self.USE_PEPTIDE_FEATURES:
            raise ValueError(
                "USE_PROTEIN_FEATURES=True requires USE_PEPTIDE_FEATURES=True."
            )
        if not any(
            (
                self.USE_PEPTIDE_FEATURES,
                self.USE_PROTEIN_FEATURES,
                self.USE_BIOLOGICAL_FEATURES,
            )
        ):
            raise ValueError("At least one feature group must be enabled.")
        if not isinstance(self.PEPTIDE_FEATURE_TYPE, str):
            raise ValueError("PEPTIDE_FEATURE_TYPE must be a string.")
        if self.PEPTIDE_FEATURE_TYPE.lower() not in {"aac", "aac_dipeptide"}:
            raise ValueError(
                "PEPTIDE_FEATURE_TYPE must be 'aac' or 'aac_dipeptide'."
            )
        self.PEPTIDE_FEATURE_TYPE = self.PEPTIDE_FEATURE_TYPE.lower()
        if not 0 < self.VALIDATION_SIZE < 1:
            raise ValueError("VALIDATION_SIZE must be between 0 and 1.")
        if self.SMOTE_K_NEIGHBORS < 1:
            raise ValueError("SMOTE_K_NEIGHBORS must be positive.")
        if self.MAX_PEPTIDE_LENGTH < 1:
            raise ValueError("MAX_PEPTIDE_LENGTH must be positive.")
        if not 0 < self.PREDICTION_THRESHOLD < 1:
            raise ValueError("PREDICTION_THRESHOLD must be between 0 and 1.")
        if self.RANDOM_FOREST_TREES < 1:
            raise ValueError("RANDOM_FOREST_TREES must be positive.")