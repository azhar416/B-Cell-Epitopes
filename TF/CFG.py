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
        self.CNN_CHANNELS = 128  # Try 32/64/128 for lower/higher CNN capacity.
        self.CNN_DROPOUT = 0.3
        self.CLASSIFIER_HIDDEN_SIZE = None  # Try 128/256/512; None uses half the encoder width.
        self.CLASSIFIER_DROPOUT = 0.5  # Try 0.3/0.5/0.7.
        self.CROSS_ATTENTION_HEADS = 8
        self.CROSS_ATTENTION_DROPOUT = 0.1
        self.FEATURE_FUSION = "cross_attention"  # sequence_only, weighted_concat, cross_attention, se, gmu, film, late_fusion, low_rank_bilinear, gated_residual
        self.FEATURE_FUSION_DROPOUT = 0.1
        self.LATE_FUSION_DROPOUT = 0.5
        self.LOW_RANK_BILINEAR_DIM = 32
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

    def validate(self):
        if not isinstance(self.FEATURE_FUSION, str) or self.FEATURE_FUSION.lower() not in {
            "sequence_only",
            "weighted_concat",
            "cross_attention",
            "se",
            "gmu",
            "film",
            "late_fusion",
            "low_rank_bilinear",
            "gated_residual",
        }:
            raise ValueError(
                "FEATURE_FUSION must be sequence_only, weighted_concat, "
                "cross_attention, se, gmu, film, late_fusion, "
                "low_rank_bilinear, or gated_residual."
            )
        self.FEATURE_FUSION = self.FEATURE_FUSION.lower()
        if (
            not isinstance(self.LOW_RANK_BILINEAR_DIM, int)
            or isinstance(self.LOW_RANK_BILINEAR_DIM, bool)
            or self.LOW_RANK_BILINEAR_DIM < 1
        ):
            raise ValueError("LOW_RANK_BILINEAR_DIM must be a positive integer.")
        if self.CLASSIFIER_TYPE.lower() not in {
            "pooled_mlp",
            "cnn",
            "attention_pooling",
            "bigru",
        }:
            raise ValueError(
                "CLASSIFIER_TYPE must be pooled_mlp, cnn, attention_pooling, or bigru."
            )
        if self.CNN_CHANNELS != "legacy" and (
            not isinstance(self.CNN_CHANNELS, int)
            or isinstance(self.CNN_CHANNELS, bool)
            or self.CNN_CHANNELS < 1
        ):
            raise ValueError("CNN_CHANNELS must be a positive integer or 'legacy'.")
        if not 0 <= self.CNN_DROPOUT < 1:
            raise ValueError("CNN_DROPOUT must be in [0, 1).")
        if self.CLASSIFIER_HIDDEN_SIZE is not None and (
            not isinstance(self.CLASSIFIER_HIDDEN_SIZE, int)
            or isinstance(self.CLASSIFIER_HIDDEN_SIZE, bool)
            or self.CLASSIFIER_HIDDEN_SIZE < 1
        ):
            raise ValueError(
                "CLASSIFIER_HIDDEN_SIZE must be a positive integer or None."
            )
        if not 0 <= self.CLASSIFIER_DROPOUT < 1:
            raise ValueError("CLASSIFIER_DROPOUT must be in [0, 1).")
        if (
            not isinstance(self.CROSS_ATTENTION_HEADS, int)
            or isinstance(self.CROSS_ATTENTION_HEADS, bool)
            or self.CROSS_ATTENTION_HEADS < 1
        ):
            raise ValueError("CROSS_ATTENTION_HEADS must be a positive integer.")
        if (
            not isinstance(self.CROSS_ATTENTION_DROPOUT, (int, float))
            or isinstance(self.CROSS_ATTENTION_DROPOUT, bool)
            or not 0 <= self.CROSS_ATTENTION_DROPOUT < 1
        ):
            raise ValueError("CROSS_ATTENTION_DROPOUT must be in [0, 1).")
        if (
            not isinstance(self.FEATURE_FUSION_DROPOUT, (int, float))
            or isinstance(self.FEATURE_FUSION_DROPOUT, bool)
            or not 0 <= self.FEATURE_FUSION_DROPOUT < 1
        ):
            raise ValueError("FEATURE_FUSION_DROPOUT must be in [0, 1).")
        if (
            not isinstance(self.LATE_FUSION_DROPOUT, (int, float))
            or isinstance(self.LATE_FUSION_DROPOUT, bool)
            or not 0 <= self.LATE_FUSION_DROPOUT < 1
        ):
            raise ValueError("LATE_FUSION_DROPOUT must be in [0, 1).")

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