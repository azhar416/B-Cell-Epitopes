import pandas as pd


class CFG:
    def __init__(self):
        self.DEBUG = False
        self.SEED = 42
        self.DATASET_NAME = "bcell"
        self._DATASET_PATH = "/kaggle/input/datasets/daffamazhar/b-cell-epitope"
        self._BCELL_DATASET_PATH = f"{self._DATASET_PATH}/input_bcell.csv"
        self._SARS_DATASET_PATH = f"{self._DATASET_PATH}/input_sars.csv"
        self._SARS2_DATASET_PATH = f"{self._DATASET_PATH}/input_covid.csv"

        self.MODEL_NAME = "Rostlab/prot_bert"
        self.DROPOUT = 0.3
        self.MAX_LENGTH = 32
        self.USE_MULTI_GPU = True
        self.FREEZE_ENCODER = True # True to freeze ProtBert encoder, False to fine-tune it
        self.USE_LORA = True # True to use LoRA, False to not use it
        self.LORA_RANK = 8
        self.LORA_ALPHA = 16
        self.LORA_DROPOUT = 0.1
        self.LORA_TARGET_MODULES = ("query", "value")

        self.TEST_SIZE = 0.2
        self.BATCH_SIZE = 32
        self.EPOCHS = 3 if self.DEBUG else 100
        self.ENCODER_LEARNING_RATE = 0.0 if self.FREEZE_ENCODER else 2e-5
        self.LORA_LEARNING_RATE = 2e-4
        self.HEAD_LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 0.001
        self.LOSS_FUNCTION = "binary_focal"  # "binary_focal" or "bce"
        self.FOCAL_ALPHA = 0.75
        self.FOCAL_GAMMA = 2.0
        self.PREDICTION_THRESHOLD = 0.5
        self.GRADIENT_CLIP_NORM = 1.0
        self.NUM_WORKERS = 0
        self.CHECKPOINT_PATH = "checkpoints/protbert_classifier.pt"

    def set_debug(self, debug=True):
        self.DEBUG = debug
        self.EPOCHS = 3 if self.DEBUG else 100

    def load_bcell_dataset(self):
        return pd.read_csv(self._BCELL_DATASET_PATH)

    def load_sars_dataset(self):
        return pd.read_csv(self._SARS_DATASET_PATH)

    def load_sars2_dataset(self):
        return pd.read_csv(self._SARS2_DATASET_PATH)