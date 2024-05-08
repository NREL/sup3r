"""Utilities used across preprocessing modules."""

import numpy as np


def get_handler_weights(data_handlers):
    """Get weights used to sample from different data handlers based on
    relative sizes"""
    sizes = [dh.size for dh in data_handlers]
    weights = sizes / np.sum(sizes)
    weights = weights.astype(np.float32)
    return weights
