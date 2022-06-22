# -*- coding: utf-8 -*-
"""Sup3r Model Software"""
from .base import Sup3rGan
from .modified_loss import (Sup3rGanCoarseMse, Sup3rGanMmdMse, Sup3rGanMmd,
                            Sup3rGanMseExp, Sup3rGanExp, Sup3rGanMae)
from .data_centric import (Sup3rGanDC, Sup3rGanDCwithExp, Sup3rGanDCwithMmdMse,
                           Sup3rGanDCwithMseExp, Sup3rGanDCwithMae)
