# -*- coding: utf-8 -*-
"""Bias correction transformation functions."""


def bc_scalar_adder(input, scalar, adder):
    """Bias correct data using a simple global *scalar +adder method.

    Parameters
    ----------
    input : np.ndarray
        Any data to be bias corrected
    scalar : float
        Scalar (multiplicative) value to apply to input data.
    adder : float
        Adder value to apply to input data.

    Returns
    -------
    out : np.ndarray
        out = input * scalar + adder
    """
    return input * scalar + adder
