from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CFG:
    def __init__(self):
        self.DEBUG = False
        self.SEED = 42
        self.DATASET_NAME = "bcell"
        self.DATA_DIR = PROJECT_ROOT / "dataset"
        self._DATASET_PATH = str(self.DATA_DIR)
        self._BCELL_DATASET_PATH = self.DATA_DIR / "input_bcell.csv"
        self._SARS_DATASET_PATH = self.DATA_DIR / "input_sars.csv"
        self._SARS2_DATASET_PATH = self.DATA_DIR / "input_covid.csv"
        self.MAX_PEPTIDE_LENGTH = 20
        self.USE_PROTEIN_FEATURES = True  # Set True to enable flanking protein context (Option B)
        self.FLANKING_SIZE = 5  # Number of residues on N- and C-terminal flanks
        self.USE_DIPEPTIDE_FEATURES = False  # Add 400-dimensional peptide DPC features.

        self.MODEL_TYPE = "esm2"  # choose "protbert", "esm2", or "prott5"
        self.CLASSIFIER_TYPE = "pooled_mlp"  # pooled_mlp, cnn, attention_pooling, bigru
        self.CNN_CHANNELS = 128  # Set to "legacy" for max(16, hidden_size // 4).
        self.CNN_DROPOUT = 0.3
        self.FEATURE_FUSION = "cross_attention"  # choose one fusion: sequence_only, weighted_concat, cross_attention, se, gmu
        self.MSD_ENABLED = False  # Independent regularizer; enable only for an MSD experiment.
        self.MSD_SAMPLES = 5
        self.MSD_DROPOUT = 0.3
        self.USE_SMOTE = True
        self.SMOTE_K_NEIGHBORS = 5
        self.PROTBERT_MODEL_NAME = "Rostlab/prot_bert"
        self.ESM2_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
        self.PROTT5_MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
        self.PROTT5_TOKENIZER_NAME = "Rostlab/prot_t5_xl_uniref50"
        self.PROTT5_LORA_TARGET_MODULES = ("q", "v")
        self.MAX_LENGTH = None  # Derived from peptide length and tokenizer special tokens.
        self.USE_MULTI_GPU = True
        self.FREEZE_ENCODER = True  # Freeze the pretrained encoder during linear probing
        self.USE_LORA = True  # Set True to train LoRA adapters instead
        self.LORA_RANK = 4
        self.LORA_ALPHA = 8
        self.LORA_DROPOUT = 0.2
        self.LORA_TARGET_MODULES = ("query", "value")

        self.VALIDATION_SIZE = 0.2
        self.BATCH_SIZE = 32
        self.EPOCHS = 3 if self.DEBUG else 100
        self.EARLY_STOPPING_PATIENCE = 10
        self.EARLY_STOPPING_MIN_DELTA = 0.001
        self.SCHEDULER_PATIENCE = 3
        self.SCHEDULER_FACTOR = 0.5
        self.SCHEDULER_MIN_LR = 1e-7
        self.ENCODER_LEARNING_RATE = 0.0 if self.FREEZE_ENCODER else 2e-5
        self.LORA_LEARNING_RATE = 5e-6
        self.HEAD_LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 0.1
        self.LABEL_SMOOTHING = 0.05
        self.LOSS_FUNCTION = "binary_focal"  # "binary_focal" or "bce"
        self.FOCAL_ALPHA = 0.75
        self.FOCAL_GAMMA = 2.0
        self.PREDICTION_THRESHOLD = 0.5
        self.GRADIENT_CLIP_NORM = 1.0
        self.NUM_WORKERS = 0
        self.CHECKPOINT_PATH = str(
            PROJECT_ROOT / "DL" / "artifacts" / "protbert_classifier.pt"
        )

    @property
    def MODEL_NAME(self):
        """Return the pretrained model matching the selected model type."""
        model_names = {
            "protbert": self.PROTBERT_MODEL_NAME,
            "esm2": self.ESM2_MODEL_NAME,
            "prott5": self.PROTT5_MODEL_NAME,
        }
        try:
            return model_names[self.MODEL_TYPE.lower()]
        except KeyError as error:
            raise ValueError(
                "MODEL_TYPE must be 'protbert', 'esm2', or 'prott5'."
            ) from error

    @property
    def EFFECTIVE_PEPTIDE_LENGTH(self):
        """Return maximum sequence length accounting for optional flanking residues."""
        if self.USE_PROTEIN_FEATURES:
            return self.MAX_PEPTIDE_LENGTH + 2 * self.FLANKING_SIZE
        return self.MAX_PEPTIDE_LENGTH

    def set_debug(self, debug=True):
        self.DEBUG = debug
        self.EPOCHS = 3 if self.DEBUG else 100

    def load_bcell_dataset(self):
        return pd.read_csv(self._BCELL_DATASET_PATH)

    def load_sars_dataset(self):
        return pd.read_csv(self._SARS_DATASET_PATH)

    def load_sars2_dataset(self):
        return pd.read_csv(self._SARS2_DATASET_PATH)