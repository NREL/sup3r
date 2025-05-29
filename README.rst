Welcome to SUP3R!

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
check out the sup3r command line interface `(CLI)
<https://nrel.github.io/sup3r/_cli/sup3r.html#sup3r>`__.

Installing sup3r
================

NOTE: The installation instruction below assume that you have python installed
on your machine and are using `conda <https://docs.conda.io/en/latest/index.html>`__
as your package/environment manager.

Option 1: Install from PIP (recommended for analysts):
------------------------------------------------------

1. Create a new environment: ``conda create --name sup3r python=3.11``

2. Activate environment: ``conda activate sup3r``

3. Install sup3r: ``pip install NREL-sup3r``

4. Run this if you want to train models on GPUs: ``pip install tensorflow[and-cuda]``

   4.1 For OSX use instead: ``python -m pip install tensorflow-metal``

Option 2: Clone repo (recommended for developers)
-------------------------------------------------

1. from home dir, ``git clone git@github.com:NREL/sup3r.git``

2. Create ``sup3r`` environment and install package
    1) Create a conda env with python: ``conda create --name sup3r python=3.11``
    2) Run the command: ``conda activate sup3r``
    3) ``cd`` into the repo cloned in 1.
    4) Prior to running ``pip`` below, make sure the branch is correct (install
       from main!)
    5) Install ``sup3r`` and its dependencies by running:
       ``pip install .`` (or ``pip install -e .`` if running a dev branch
       or working on the source code)
    6) Run this if you want to train models on GPUs: ``pip install tensorflow[and-cuda]``
    7) *Optional*: Set up the pre-commit hooks with ``pip install pre-commit`` and ``pre-commit install``

Recommended Citation
====================

Update with current version and DOI:

Brandon Benton, Grant Buster, Guilherme Pimenta Castelao, Malik Hassanaly, Pavlo Pinchuk, Slater Podgorny, Andrew Glaws, and Ryan King. Super Resolution for Renewable Resource Data (sup3r). https://github.com/NREL/sup3r (version v0.2.0), 2024. DOI: 10.5281/zenodo.14042894

Acknowledgments
===============

This work was authored by the National Renewable Energy Laboratory, operated for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This research was supported by the Grid Modernization Initiative of the U.S. Department of Energy (DOE) as part of its Grid Modernization Laboratory Consortium, a strategic partnership between DOE and the national laboratories to bring together leading experts, technologies, and resources to collaborate on the goal of modernizing the nation’s grid. Funding provided by the the DOE Office of Energy Efficiency and Renewable Energy (EERE), the DOE Office of Electricity (OE), DOE Grid Deployment Office (GDO), the DOE Office of Fossil Energy and Carbon Management (FECM), and the DOE Office of Cybersecurity, Energy Security, and Emergency Response (CESER), the DOE Advanced Scientific Computing Research (ASCR) program, the DOE Solar Energy Technologies Office (SETO), the DOE Wind Energy Technologies Office (WETO), the United States Agency for International Development (USAID), and the Laboratory Directed Research and Development (LDRD) program at the National Renewable Energy Laboratory. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
