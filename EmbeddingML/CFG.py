"""Configuration for frozen ProtT5 embeddings with classical classifiers."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    DATASET_NAME: str = "bcell"
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    OUTPUT_DIR: Path = field(
        default_factory=lambda: PROJECT_ROOT / "EmbeddingML" / "artifacts"
    )
    MAX_PEPTIDE_LENGTH: int = 20
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    ENCODER_NAME: str = "Rostlab/prot_t5_xl_half_uniref50-enc"
    TOKENIZER_NAME: str = "Rostlab/prot_t5_xl_uniref50"
    BATCH_SIZE: int = 8
    DEVICE: str = "auto"
    USE_BIOLOGICAL_FEATURES: bool = False
    FORCE_RECOMPUTE_EMBEDDINGS: bool = False

    MODEL_TYPE: str = "svm"
    SVM_C: float = 10.0
    SVM_GAMMA: str | float = "scale"
    SVM_CLASS_WEIGHT: str | None = "balanced"
    RANDOM_FOREST_TREES: int = 500
    RANDOM_FOREST_MAX_DEPTH: int | None = None
    RANDOM_FOREST_MIN_SAMPLES_LEAF: int = 2
    RANDOM_FOREST_CLASS_WEIGHT: str | None = "balanced"
    PREDICTION_THRESHOLD: float = 0.5

    def validate(self) -> None:
        self.DATASET_NAME = self.DATASET_NAME.lower()
        if self.DATASET_NAME not in {"bcell", "sars"}:
            raise ValueError("DATASET_NAME must be 'bcell' or 'sars'.")
        self.MODEL_TYPE = self.MODEL_TYPE.lower()
        if self.MODEL_TYPE not in {"svm", "random_forest"}:
            raise ValueError("MODEL_TYPE must be 'svm' or 'random_forest'.")
        if self.MAX_PEPTIDE_LENGTH < 1:
            raise ValueError("MAX_PEPTIDE_LENGTH must be positive.")
        if not 0 < self.VALIDATION_SIZE < 1:
            raise ValueError("VALIDATION_SIZE must be between 0 and 1.")
        if self.BATCH_SIZE < 1:
            raise ValueError("BATCH_SIZE must be positive.")
        if self.DEVICE not in {"auto", "cpu"} and not self.DEVICE.startswith("cuda"):
            raise ValueError("DEVICE must be 'auto', 'cpu', or a CUDA device.")
        if not isinstance(self.USE_BIOLOGICAL_FEATURES, bool):
            raise ValueError("USE_BIOLOGICAL_FEATURES must be a boolean.")
        if not isinstance(self.FORCE_RECOMPUTE_EMBEDDINGS, bool):
            raise ValueError("FORCE_RECOMPUTE_EMBEDDINGS must be a boolean.")
        if self.SVM_C <= 0:
            raise ValueError("SVM_C must be positive.")
        if not isinstance(self.SVM_GAMMA, (str, float, int)):
            raise ValueError("SVM_GAMMA must be 'scale', 'auto', or a positive number.")
        if isinstance(self.SVM_GAMMA, str) and self.SVM_GAMMA not in {
            "scale",
            "auto",
        }:
            raise ValueError("SVM_GAMMA must be 'scale', 'auto', or a positive number.")
        if isinstance(self.SVM_GAMMA, (float, int)) and self.SVM_GAMMA <= 0:
            raise ValueError("Numeric SVM_GAMMA must be positive.")
        if self.RANDOM_FOREST_TREES < 1:
            raise ValueError("RANDOM_FOREST_TREES must be positive.")
        if self.RANDOM_FOREST_MIN_SAMPLES_LEAF < 1:
            raise ValueError("RANDOM_FOREST_MIN_SAMPLES_LEAF must be positive.")
        if self.RANDOM_FOREST_MAX_DEPTH is not None and self.RANDOM_FOREST_MAX_DEPTH < 1:
            raise ValueError("RANDOM_FOREST_MAX_DEPTH must be positive or None.")
        if not 0 < self.PREDICTION_THRESHOLD < 1:
            raise ValueError("PREDICTION_THRESHOLD must be between 0 and 1.")
