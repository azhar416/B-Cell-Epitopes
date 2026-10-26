from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    """Configuration for preprocessing, stratified validation, and training."""

    DATASET_NAME: str = "bcell"
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    OUTPUT_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "ML" / "artifacts")
    MODEL_TYPE: str = "logistic_regression"
    FEATURE_SET: str = "aac"
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
        if self.FEATURE_SET.lower() not in {"aac", "aac_dipeptide"}:
            raise ValueError("FEATURE_SET must be 'aac' or 'aac_dipeptide'.")
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