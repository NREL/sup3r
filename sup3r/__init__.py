# -*- coding: utf-8 -*-
"""Super Resolving Renewable Energy Resource Data (SUP3R)"""
import os
from sup3r.version import __version__
# Next import sets up CLI commands
# This line could be "import sup3r.cli" but that breaks sphinx as of 12/11/2023
from sup3r.cli import main

__author__ = """Brandon Benton"""
__email__ = "brandon.benton@nrel.gov"

SUP3R_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_DIR = os.path.join(SUP3R_DIR, 'configs')
TEST_DATA_DIR = os.path.join(os.path.dirname(SUP3R_DIR), 'tests', 'data')
