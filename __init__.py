# thalassa/__init__.py

"""
🚀 A neural network built from scratch using Python, supporting dense layers, dropout, ReLU, Leaky ReLU, tanh, sigmoid, softmax, cross-entropy, binary cross-entropy, and SGD. More features like Adam, RMSprop, and sparse categorical cross-entropy coming soon! 
"""

__version__ = "0.1"

import importlib


def __getattr__(name):
    try:
        return importlib.import_module("." + name, __name__)
    except ImportError:
        raise AttributeError(f"Module 'thalassa' has no attribute '{name}'")
