# -*- coding: utf-8 -*-
"""Sup3r models with modified content loss functions"""


from sup3r.models.base import Sup3rGan
from sup3r.utilities.loss_metrics import ExpLoss, MmdMseLoss


class Sup3rGanMmdMse(Sup3rGan):
    """Sup3rGan sub class using mse + max mean discrepancy for content
    loss instead of just mse"""

    LOSS = MmdMseLoss()


class Sup3rGanExp(Sup3rGan):
    """Sup3rGan sub class using mse + exponential difference for content
    loss instead of just mse"""

    LOSS = ExpLoss()
