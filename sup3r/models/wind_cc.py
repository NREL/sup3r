# -*- coding: utf-8 -*-
"""Sup3r model software"""
import logging
import tensorflow as tf

from sup3r.models.base import Sup3rGan


logger = logging.getLogger(__name__)


class SolarCC(Sup3rGan):
    """Wind climate change model.

    Modifications to standard Sup3rGan:
        - Hi res topography is expected as the last feature channel in the true
          data in the true batch observation. This topo channel is appended to
          the generated output so the discriminator can look at the wind fields
          compared to the associated hi res topo.
    """

