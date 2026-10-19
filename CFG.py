from pandas import pd

class CFG:
    def __init__(self):
        self.SEED = 42
        self._DATASET_PATH = "/kaggle/input/datasets/daffamazhar/b-cell-epitope"
        self._BCELL_DATASET_PATH = f"{self._DATASET_PATH}/input_bcell.csv"
        self._SARS_DATASET_PATH = f"{self._DATASET_PATH}/input_sars.csv"
        self._SARS2_DATASET_PATH = f"{self._DATASET_PATH}/input_covid.csv"

    def load_bcell_dataset(self):
        return pd.read_csv(self._BCELL_DATASET_PATH)

    def load_sars_dataset(self):
        return pd.read_csv(self._SARS_DATASET_PATH)

    def load_sars2_dataset(self):
        return pd.read_csv(self._SARS2_DATASET_PATH)