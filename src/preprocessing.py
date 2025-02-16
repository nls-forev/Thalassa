import numpy as np

def OneHotEncoder(y: np.ndarray) -> np.ndarray:
    num_classes = np.max(y) + 1
    y_encoded = np.eye(num_classes)[
        y.squeeze()]

    return y_encoded
