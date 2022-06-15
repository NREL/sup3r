# -*- coding: utf-8 -*-
"""Sup3r models with modified content loss functions"""


from sup3r.models.base import Sup3rGan
from sup3r.utilities.loss_metrics import (ExpLoss, MmdMseLoss, MmdLoss,
                                          CoarseMseLoss)


class Sup3rGanMmd(Sup3rGan):
    """Sup3rGan sub class using only max mean discrepancy as the content loss
    """
    LOSS = MmdLoss()


class Sup3rGanMmdMse(Sup3rGan):
    """Sup3rGan sub class using mse + max mean discrepancy for content
    loss instead of just mse"""
    LOSS = MmdMseLoss()


class Sup3rGanExp(Sup3rGan):
    """Sup3rGan sub class using exponential difference for content loss instead
    of mse"""

    LOSS = ExpLoss()


class Sup3rGanCoarseMse(Sup3rGan):
    """Sup3rGan sub class using coarse MSE on the average values of the spatial
    fields instead of an element-to-element comparison."""
    LOSS = CoarseMseLoss()
