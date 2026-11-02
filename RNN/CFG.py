from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    """Configuration for the isolated BiLSTM peptide classifier."""

    DATASET_NAME: str = "bcell"
    DATA_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "dataset")
    OUTPUT_DIR: Path = field(default_factory=lambda: PROJECT_ROOT / "RNN" / "artifacts")
    MAX_PEPTIDE_LENGTH: int = 20
    FLANKING_SIZE: int = 5
    VALIDATION_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EMBEDDING_DIM: int = 64
    HIDDEN_SIZE: int = 128
    NUM_LAYERS: int = 1
    DROPOUT: float = 0.3
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.001
    PREDICTION_THRESHOLD: float = 0.5
    NUM_WORKERS: int = 0
    DEVICE: str = "auto"

    def validate(self) -> None:
        if self.DATASET_NAME.lower() not in {"bcell", "sars"}:
            raise ValueError("DATASET_NAME must be 'bcell' or 'sars'.")
        if self.MAX_PEPTIDE_LENGTH < 1:
            raise ValueError("MAX_PEPTIDE_LENGTH must be positive.")
        if self.FLANKING_SIZE < 0:
            raise ValueError("FLANKING_SIZE must be non-negative.")
        if not 0 < self.VALIDATION_SIZE < 1:
            raise ValueError("VALIDATION_SIZE must be between 0 and 1.")
        if self.BATCH_SIZE < 1 or self.EPOCHS < 1:
            raise ValueError("BATCH_SIZE and EPOCHS must be positive.")
        if self.EMBEDDING_DIM < 1 or self.HIDDEN_SIZE < 1:
            raise ValueError("EMBEDDING_DIM and HIDDEN_SIZE must be positive.")
        if self.NUM_LAYERS < 1:
            raise ValueError("NUM_LAYERS must be positive.")
        if not 0 <= self.DROPOUT < 1:
            raise ValueError("DROPOUT must be in [0, 1).")
        if self.LEARNING_RATE <= 0 or self.WEIGHT_DECAY < 0:
            raise ValueError("LEARNING_RATE must be positive and WEIGHT_DECAY non-negative.")
        if not 0 < self.PREDICTION_THRESHOLD < 1:
            raise ValueError("PREDICTION_THRESHOLD must be between 0 and 1.")