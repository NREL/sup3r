#################
Welcome to SUP3R!
#################

.. image:: https://github.com/NREL/sup3r/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/sup3r/

.. image:: https://github.com/NREL/sup3r/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/sup3r/actions?query=workflow%3A%22Pytests%22

.. image:: https://github.com/NREL/sup3r/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/NREL/sup3r/actions?query=workflow%3A%22Lint+Code+Base%22

.. image:: https://img.shields.io/pypi/pyversions/NREL-sup3r.svg
    :target: https://pypi.org/project/NREL-sup3r/

.. image:: https://badge.fury.io/py/NREL-sup3r.svg
    :target: https://badge.fury.io/py/NREL-sup3r

.. image:: https://codecov.io/gh/nrel/sup3r/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/nrel/sup3r

.. image:: https://zenodo.org/badge/422324608.svg
    :target: https://zenodo.org/badge/latestdoi/422324608

.. inclusion-intro

The Super Resolution for Renewable Resource Data (sup3r) software uses
generative adversarial networks to create synthetic high-resolution wind and
solar spatiotemporal data from coarse low-resolution inputs. To get started,
check out the sup3r command line interface (CLI) `here
<https://nrel.github.io/sup3r/_cli/sup3r.html#sup3r>`_.

Installing sup3r
================

NOTE: The installation instruction below assume that you have python installed
on your machine and are using `conda <https://docs.conda.io/en/latest/index.html>`_
as your package/environment manager.

Option 1: Install from PIP or Conda (recommended for analysts):
---------------------------------------------------------------

1. Create a new environment: ``conda create --name sup3r python=3.9``

2. Activate directory: ``conda activate sup3r``

3. Install sup3r: ``pip install NREL-sup3r``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com:NREL/sup3r.git``

2. Create ``sup3r`` environment and install package
    1) Create a conda env: ``conda create -n sup3r``
    2) Run the command: ``conda activate sup3r``
    3) cd into the repo cloned in 1.
    4) prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``sup3r`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

Recommended Citation
====================

Update with current version and DOI:

Brandon Benton, Grant Buster, Andrew Glaws, Ryan King. Super Resolution for Renewable Resource Data (sup3r). https://github.com/NREL/sup3r (version v0.0.0), 2022. DOI: 10.5281/zenodo.6808547
