import pandas as pd


class CFG:
    def __init__(self):
        self.SEED = 42
        self.DATASET_NAME = "bcell"
        self._DATASET_PATH = "/kaggle/input/datasets/daffamazhar/b-cell-epitope"
        self._BCELL_DATASET_PATH = f"{self._DATASET_PATH}/input_bcell.csv"
        self._SARS_DATASET_PATH = f"{self._DATASET_PATH}/input_sars.csv"
        self._SARS2_DATASET_PATH = f"{self._DATASET_PATH}/input_covid.csv"

        self.MODEL_NAME = "Rostlab/prot_bert"
        self.DROPOUT = 0.3
        self.MAX_LENGTH = 32

        self.TEST_SIZE = 0.2
        self.BATCH_SIZE = 16
        self.EPOCHS = 3
        self.ENCODER_LEARNING_RATE = 2e-5
        self.HEAD_LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 0.01
        self.FOCAL_ALPHA = 0.75
        self.FOCAL_GAMMA = 2.0
        self.PREDICTION_THRESHOLD = 0.5
        self.GRADIENT_CLIP_NORM = 1.0
        self.NUM_WORKERS = 0
        self.CHECKPOINT_PATH = "checkpoints/protbert_classifier.pt"

    def load_bcell_dataset(self):
        return pd.read_csv(self._BCELL_DATASET_PATH)

    def load_sars_dataset(self):
        return pd.read_csv(self._SARS_DATASET_PATH)

    def load_sars2_dataset(self):
        return pd.read_csv(self._SARS2_DATASET_PATH)