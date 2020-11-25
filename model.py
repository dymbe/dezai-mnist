import numpy as np


class Model:
    def train(self, data) -> None:
        pass

    def test_score(self, data) -> float:
        pass

    def outputs(self, test_loader) -> np.ndarray:
        pass
