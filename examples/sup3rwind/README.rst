###################
Sup3rWind Examples
###################

Super-Resolution for Renewable Energy Resource Data with Wind from Reanalysis Data (Sup3rWind) is one application of the sup3r software. In this work, we train generative models to create high-resolution (2km 5-minute) wind data based on coarse (30km hourly) ERA5 data. The generative models and high-resolution output data is publicly available via the `Open Energy Data Initiative (OEDI) <https://data.openei.org/s3_viewer?bucket=nrel-pds-wtk&prefix=sup3rwind%2F>`_ and via HSDS at the bucket ``nrel-pds-hsds`` and path ``/nrel/wtk/sup3rwind``. This data covers recent historical time periods for an expanding selection of countries.

Sup3rWind Data Access
----------------------

The Sup3rWind data and models are publicly available in a public AWS S3 bucket. The data files can be downloaded directly from there to your local machine or an EC2 instance using the `OEDI data explorer <https://data.openei.org/s3_viewer?bucket=nrel-pds-wtk&prefix=sup3rwind%2F>`_ or the `AWS CLI <https://aws.amazon.com/cli/>`_. A word of caution: there's a lot of data here. The smallest Sup3rWind file for just a single variable at 2-km 5-minute resolution is 130 GB.

The Sup3rWind data is also loaded into `HSDS <https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/>`_ so that you may stream the data via the `NREL developer API <https://developer.nrel.gov/signup/>`_ or your own HSDS server. This is the best option if you're not going to want a full annual dataset. See these `rex instructions <https://nrel.github.io/rex/misc/examples.hsds.html>`_ for more details on how to access this data with HSDS and rex.

Example Sup3rWind Data Usage
-----------------------------

Sup3rWind data can be used in generally the same way as Sup3rCC data, with the condition that Sup3rWind includes only wind data and ancillary variables for modeling wind energy generation. Refer to the Sup3rCC example notebook `here <https://github.com/NREL/sup3r/tree/main/examples/sup3rcc/using_the_data.ipynb>`_ for usage patterns.

Running Sup3rWind Models
-------------------------

The process for running the Sup3rWind models is much the same as for Sup3rCC (``sup3r/examples/sup3rcc/README.rst``).

#. Download the Sup3rWind models to your hardware using the AWS CLI: ``$ aws s3 cp s3://nrel-pds-wtk/sup3rwind/models/``
#. Download the ERA5 data that you want to downscale from `ERA5-single-levels <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview/>`_ and/or `ERA5-pressure-levels <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview/>`_.
#. Setup the Sup3rWind software. We recommend using `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your python environments. You can create a sup3r environment with the conda file in this example directory: ``$ conda env create -n sup3rwind --file env.yml``
#. Copy this examples directory to your hardware. You're going to be using the folder structure in ``/sup3r/examples/sup3rwind/run_configs`` as your project directories (``/sup3r/`` is a git clone of the sup3r software repo).
#. Navigate to ``/sup3r/examples/sup3rwind/run_configs/wind/`` and/or ``sup3r/examples/sup3rwind/run_configs/trhp`` and update all of the filepaths in the config files for the source ERA5 data, Sup3rWind models, and exogenous data sources (e.g. the ``topography`` source file).
#. Update the execution control parameters in the ``config_fwp_spatial.json`` file based on the hardware you're running on.
#. Run ``sup3r-pipeline`` to run just one job. There are also batch options for running multiple jobs, but we recommend starting with ``sup3r-pipeline`` (more on the sup3r CLIs `here <https://nrel.github.io/sup3r/_cli/sup3r.html>`_).
#. To run ``sup3r-pipeline``, make sure you are in the directory with the ``config_pipeline.json`` and ``config_fwp_spatial.json`` files, and then run this command: ``python -m sup3r.cli -c config_pipeline.json pipeline``
#. If you're running on a slurm cluster, this will kick off a number of jobs that you can see with the ``squeue`` command. If you're running locally, your terminal should now be running the Sup3rWind models. The software will create a ``./logs/`` directory in which you can monitor the progress of your jobs.
#. The ``sup3r-pipeline`` is designed to run several modules in serial, with each module running multiple chunks in parallel. Once the first module (forward-pass) finishes, you'll want to run ``python -m sup3r.cli -c config_pipeline.json pipeline`` again. This will clean up status files and kick off the next step in the pipeline (if the current step was successful).

Sup3rWind Versions
-------------------

The Sup3rWind data has versions that coincide with the sup3r software versions. Note that not every sup3r software version will have a corresponding Sup3rWind data release, but every Sup3rWind data release will have a corresponding sup3r software version.

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Version
      - Effective Date
      - Notes
    * - 0.1.2
      - 3/15/2024
      - Initial release of Sup3rWind for Ukraine, Moldova, and part of Romania. Includes 2-km 5-minute wind speed and wind direction data and 2-km hourly wind speed, wind direction, pressure, temperature, and relative humidity data for 2000-2023.


Recommended Citation
---------------------

Brandon N. Benton, Grant Buster, Pavlo Pinchuk, Andrew Glaws, Ryan N. King, Galen Maclaurin, Ilya Chernyakhovskiy. "Super-Resolution for Renewable Energy Resource Data with Wind from Reanalysis Data (Sup3rWind)". In Prep.

Acknowledgements
-----------------

This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the DOE Grid Deployment Office (GDO), the DOE Advanced Scientific Computing Research (ASCR) program, the DOE Solar Energy Technologies Office (SETO), and the Laboratory Directed Research and Development (LDRD) program at the National Renewable Energy Laboratory. The research was performed using computational resources sponsored by the DOE Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
