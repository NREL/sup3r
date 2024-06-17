#################
Welcome to SUP3R!
#################

|Docs| |Tests| |Linter| |PyPi| |PythonV| |Codecov| |Zenodo|

.. |Docs| image:: https://github.com/NREL/sup3r/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/sup3r/

.. |Tests| image:: https://github.com/NREL/sup3r/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/sup3r/actions?query=workflow%3A%22Pytests%22

.. |Linter| image:: https://github.com/NREL/sup3r/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/NREL/sup3r/actions?query=workflow%3A%22Lint+Code+Base%22

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/NREL-sup3r.svg
    :target: https://pypi.org/project/NREL-sup3r/

.. |PythonV| image:: https://badge.fury.io/py/NREL-sup3r.svg
    :target: https://badge.fury.io/py/NREL-sup3r

.. |Codecov| image:: https://codecov.io/gh/nrel/sup3r/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/nrel/sup3r

.. |Zenodo| image:: https://zenodo.org/badge/422324608.svg
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

3. Install sup3r: ``pip install NREL-sup3r``

4. Run this if you want to train models on GPUs: ``conda install -c anaconda tensorflow-gpu``

   4.1 For OSX use instead: ``python -m pip install tensorflow-metal``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com:NREL/sup3r.git``

2. Create ``sup3r`` environment and install package
    1) Create a conda env: ``conda create -n sup3r``
    2) Run the command: ``conda activate sup3r``
    3) ``cd`` into the repo cloned in 1.
    4) Prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``sup3r`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)
    6) Run this if you want to train models on GPUs: ``conda install -c anaconda tensorflow-gpu``
       On Eagle HPC, you will need to also run ``pip install protobuf==3.20.*`` and ``pip install chardet``
    7) *Optional*: Set up the pre-commit hooks with ``pip install pre-commit`` and ``pre-commit install``

Option 3: Using pixi (for developers)
-------------------------------------

Using pixi we have a well controlled environment which guarantee
reproducibility and avoid issues of new dependencies that can break
the tree of compatibility. While we can use the latest dependency versions for
development, we should use the 'frozen' environment for production so we can
be consistent in our results and able to traceback potential bugs. Every once
in a while we shall update the ``pixi.lock`` file, which defines the
environment.

1. Install pixi. We only need to do it once per machine. Follow the
   instructions at: ``https://pixi.sh/latest/``

2. Clone the ``sup3r`` repository: ``git clone git@github.com:NREL/sup3r.git``

3. All the configuration is in pyproject.toml, so eveyone uses the same
   environment. One important concept is that pixi try to update (localy) the
   dependency tree often. We usually want to avoid that so we are sure that we
   are running with the exact same environment, so we should add ``--frozen``.

   - To run a command with fixed environment: ``pixi run --frozen my_command``
   - To run a command with latest dependencies: ``pixi run my_command``
   - To open a terminal with default environment: ``pixi shell``
   - To open a terminal with the development extras: ``pixi shell -e dev``
   - Check if tensorflow detected the GPUs: ``pixi run check_devices``
   - To run tests in kestrel (remember that it should be in a node with GPUs):
     ``pixi run -e kestrel --frozen pytest tests/bias/test_bias_correction.py``

Recommended Citation
====================

Update with current version and DOI:

Brandon Benton, Grant Buster, Andrew Glaws, Ryan King. Super Resolution for Renewable Resource Data (sup3r). https://github.com/NREL/sup3r (version v0.0.3), 2022. DOI: 10.5281/zenodo.6808547

Acknowledgments
===============

This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the DOE Grid Deployment Office (GDO), the DOE Advanced Scientific Computing Research (ASCR) program, the DOE Solar Energy Technologies Office (SETO), the DOE Wind Energy Technologies Office (WETO), the United States Agency for International Development (USAID), and the Laboratory Directed Research and Development (LDRD) program at the National Renewable Energy Laboratory. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
