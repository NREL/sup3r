.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>
#################
TO DO
#################

#. |ss| Check output of first moment model for spatial training only |se|
#. |ss| Utilities to make make the second moment dataset |se|
    #. |ss| Validation loss calculation needs fixing |se|
#. |ss| Make the training movies |se| 
#. |ss| Add utilities to plot losses and timing |se|
#. |ss| Clean the testing suite |se|
#. |ss| Make an option to train with subfilter field or full field |se|
    #. |ss| Add upsampling functions |se|
    #. |ss| Include testing suite for mom1 |se|
    #. |ss| Include testing suite for mom2 |se|
#. |ss| Spatio temporal |se|
    #. |ss| Add training test for non subfilter training |se|
    #. |ss| Add training test for subfilter training (make sure subfilter contruction in time is appropriate) |se|
#. |ss| Make options to add arbitrary fields as input |se|
    #. |ss| Make sure we can add the other train only feature |se|
    #. |ss| Include in testing suite |se|
    #. |ss| Add `wind_condition_moments.py` to handle topography |se|
    #. |ss| Include in testing suite |se|
#. |ss| Nomenclature upsampling enhance |se|
#. |ss| See of Sup3rCondMom can instead inherit from abstract Gan class |se|
#. |ss| Add option to crop output |se|
#. Train network with increasing complexity
    #. |ss| Include number of parameter in loss plotting |se|
    #. Show training results
    

- |ss| See if we can make `conditional_moments.py` inherit from base model to avoid redundancy |se|
- Test adding tf function on the run gradient descent
- Timing of batch handler and data handler

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

Option 1: Install from PIP (recommended for analysts):
------------------------------------------------------

1. Create a new environment: ``conda create --name sup3r python=3.9``

2. Activate environment: ``conda activate sup3r``

3. Run this if you want to train models on GPUs: ``conda install -c anaconda tensorflow-gpu``

4. Install sup3r: ``pip install NREL-sup3r``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com:NREL/sup3r.git``

2. Create ``sup3r`` environment and install package
    1) Create a conda env: ``conda create -n sup3r``
    2) Run the command: ``conda activate sup3r``
    3) ``cd`` into the repo cloned in 1.
    4) Run this if you want to train models on GPUs: ``conda install -c anaconda tensorflow-gpu``
    5) Prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    6) Install ``sup3r`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)

Recommended Citation
====================

Update with current version and DOI:

Brandon Benton, Grant Buster, Andrew Glaws, Ryan King. Super Resolution for Renewable Resource Data (sup3r). https://github.com/NREL/sup3r (version v0.0.3), 2022. DOI: 10.5281/zenodo.6808547

Acknowledgments
===============

This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the DOE Office of Grid Deployment (OGD), the DOE Solar Energy Technologies Office (SETO) and USAID. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
