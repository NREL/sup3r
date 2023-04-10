################
Sup3rCC Examples
################

Super-Resolution for Renewable Energy Resource Data with Climate Change Impacts (Sup3rCC) is one application of the sup3r software. In this work, we train generative models to create high-resolution (4km hourly) wind, solar, and temperature data based on coarse (100km daily) global climate model data (GCM). The generative models and high-resolution output data are publicly available on AWS S3 at the URI ``s3://nrel-pds-sup3rcc/`` and via HSDS at the bucket ``nrel-pds-hsds`` and path ``/nrel/sup3rcc/``. This set of examples lays out basic ways to use the Sup3rCC models and data.

Sup3rCC Data Access
===================

The Sup3rCC data and models are publicly available in a public AWS S3 bucket. The data files and models can be downloaded directly from there to your local machine or an EC2 instance using the `AWS CLI <https://aws.amazon.com/cli/>`_. Be careful though, the smallest Sup3rCC file for just a single variable is 18 GB, and a full year of data is 216 GB.

The Sup3rCC data is also loaded into `HSDS <https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/>`_ so that you may stream the data via the `NREL developer API <https://developer.nrel.gov/signup/>`_ or your own HSDS server. This is the best option if you're not going to want the full annual dataset over the whole United States. See these `rex instructions <https://nrel.github.io/rex/misc/examples.hsds.html>`_ for more details on how to access this data with HSDS and rex.

Example Sup3rCC Data Usage
==========================

The jupyter notebook in this examples directory shows a basic example of how to access and explore the data. You can walk through the example notebook `here <https://github.com/NREL/sup3r/tree/main/examples/sup3rcc/using_the_data.ipynb>`_, or clone this repo, setup a basic python environment with `rex <https://github.com/NREL/rex>`_ and some plotting utilities, and run the notebook on your own.

Running Sup3rCC Models
======================

In a first-of-a-kind data product, we have released the pre-trained Sup3rCC generative machine learning models along with the sup3r software so that anyone working at the intersection of energy and climate may create their own high-resolution renewable energy resource data from GCM input. You might want to do this if you have your own GCMs or climate scenarios that you're interested in studying.

To run the Sup3rCC models, follow these instructions:

#. Decide what kind of hardware you're going to use. You could technically run Sup3rCC on a local computer, but you will need lots of RAM (we use compute nodes with 170 GB of RAM). We recommend a high-performance-computing cluster if you have access to one, or an `AWS Parallel Cluster <https://aws.amazon.com/hpc/parallelcluster/>`_ if you do not.
#. Download the Sup3rCC models to your hardware using the AWS CLI: ``$ aws s3 cp s3://nrel-pds-sup3rcc/models/``
#. Download the GCM data that you want to downscale from `CMIP6 <https://esgf-node.llnl.gov/search/cmip6/>`_
#. Setup the Sup3rCC software. We recommend using `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your python environments. You can create a sup3r environment with the conda file in this example directory: ``$ conda env create -n sup3rcc --file sup3rcc_env.yml``
#. Copy this examples directory to your hardware. You're going to be using the folder structure in ``/sup3r/examples/run_configs`` as your project directories (where ``/sup3r/`` is a git clone of this sup3r software repo.
#. Navigate to ``/sup3r/examples/run_configs/trh/`` and update all of the filepaths in the config files for the source GCM data, Sup3rCC models, and exogenous data sources (e.g. the ``nsrdb_clearsky.h5`` file).
#. Update the execution control parameters in the ``config_fwp.json`` file based on the hardware you're running on.
#. You can either run sup3r-batch to setup multiple run years, or sup3r-pipeline to run just one job. We recommend starting with sup3r-pipeline.
#. To run sup3r-pipeline, make sure you are in the directory with the ``config_pipeline.json`` and ``config_fwp.json`` files, and then run this command: ``python -m sup3r.cli -c config_pipeline.json pipeline``
#. If you're running on a slurm cluster, this will kick off a number of jobs that you can see with the ``squeue`` command. If you're running locally, your terminal should now be running the Sup3rCC models. The software will create a ``./logs/`` directory in which you can monitor the progress of your jobs.
#. The sup3r-pipeline is designed to run several modules in serial, with each module running multiple chunks in parallel. Once the first module (forward-pass) finishes, you'll want to run ``python -m sup3r.cli -c config_pipeline.json pipeline`` again. This will clean up status files and kick off the next step in the pipeline (if the current step was successful).

Recommended Citation
====================

Grant Buster, Brandon Benton, Andrew Glaws, and Ryan King. "Super-Resolution for Renewable Energy Resource Data with Climate Change Impacts using Generative Machine Learning". Under review (April 2023).
