#################
Welcome to SUP3R!
#################

The Super Resolution for Renewable Resource Data (sup3r) software uses
generative adversarial networks to create synthetic high-resolution wind and
solar spatiotemporal data from coarse low-resolution inputs.

sup3r command line tools
======================

- `forward_pass <https://nrel.github.io/sup3r/_cli/forward_pass.html#forward_pass>`_
- `data_extract <https://nrel.github.io/sup3r/_cli/data_extract.html#data_extract>`_

Installing sup3r
================

NOTE: The installation instruction below assume that you have python installed
on your machine and are using `conda <https://docs.conda.io/en/latest/index.html>`_
as your package/environment manager.

Option 1: Install from PIP or Conda (recommended for analysts):
---------------------------------------------------------------

1. Create a new environment:
    ``conda create --name sup3r``

2. Activate directory:
    ``conda activate sup3r``

3. Install rex:
    1) ``pip install NREL-sup3r`` or
    2) ``conda install nrel-sup3r --channel=nrel``

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
