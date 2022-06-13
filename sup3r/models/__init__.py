# -*- coding: utf-8 -*-
"""Sup3r Model Software"""
from .base import Sup3rGan
from .modified_loss import (Sup3rGanCoarseMse, Sup3rGanMseKld, Sup3rGanMseMmd,
                            Sup3rGanMmd, Sup3rGanKld)
from .data_centric import Sup3rGanDC, Sup3rGanDCwithKLD, Sup3rGanDCwithMMD
