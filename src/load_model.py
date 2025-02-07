from models import *
import numpy as np

def load_(path: str) -> "Sequential":
    cache = np.load(rf'{path}', allow_pickle=True)
    model = Sequential()
    model.cache = cache
    return model
