# -*- coding: utf-8 -*-
"""
Sup3r command line interface (CLI).
"""
import click
import logging

from sup3r.version import __version__
from sup3r.pipeline.forward_pass_cli import from_config as fp_cli
from sup3r.pipeline.pipeline_cli import from_config as pipe_cli
from sup3r.pipeline.pipeline_cli import valid_config_keys as pipeline_keys
from sup3r.preprocessing.data_extract_cli import from_config as dh_cli
from sup3r.postprocessing.data_collect_cli import from_config as dc_cli


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--config_file', '-c',
              required=True, type=click.Path(exists=True),
              help='sup3r configuration file json for a single module.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, config_file, verbose):
    """Sup3r command line interface.

    Try using the following commands to pull up the help pages for the
    respective sup3r CLIs::

        $ sup3r --help

        $ sup3r -c config.json pipeline --help

        $ sup3r -c config.json forward-pass --help

        $ sup3r -c config.json data-collect --help

        $ sup3r -c config.json data-extract --help

    Typically, a good place to start is to set up a sup3r job with a pipeline
    config that points to several sup3r modules that you want to run in serial.
    You would call the sup3r pipeline CLI using either of these equivelant
    commands::

        $ sup3r -c config_pipeline.json pipeline

        $ sup3r-pipeline from-config -c config_pipeline.json

    See the help pages of the module CLIs for more details on the config files
    for each CLI.
    """
    ctx.ensure_object(dict)
    ctx.obj['CONFIG_FILE'] = config_file
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def forward_pass(ctx, verbose):
    """Sup3r forward pass to super-resolve data.

    The sup3r forward pass is where all the magic happens. This module takes in
    low resolution data from source files (e.g. WRF or GCM outputs), loads a
    saved Sup3rGan model, does a forward pass of the low res data through the
    generator, and then saves the output high resolution data to disk. This
    module also handles multi-node and multi-core parallelization.

    You can call the forward-pass module via the sup3r-pipeline CLI, or call it
    directly with either of these equivelant commands::

        $ sup3r -c config_fwp.json forward-pass

        $ sup3r-forward-pass from-config -c config_fwp.json

    A sup3r forward pass config.json file can contain any arguments or keyword
    arguments required to initialize the
    :class:`sup3r.pipeline.forward_pass.ForwardPassStrategy` module. The config
    also has several optional arguments: ``log_pattern``, ``log_level``, and
    ``execution_control``. Here's a small example forward pass config::

        {
            "file_paths": "./source_files*.nc",
            "model_args": "./sup3r_model/",
            "out_pattern": "./output/sup3r_out_{file_id}.h5",
            "log_pattern": "./logs/log_{node_index}.log",
            "log_level": "DEBUG",
            "s_enhance": 20,
            "t_enhance": 24,
            "fwp_chunk_shape": [5, 5, 3],
            "spatial_overlap": 1,
            "temporal_overlap": 1,
            "max_workers": 1,
            "execution_control": {
                "option": "local"
            },
            "execution_control_eagle": {
                "option": "eagle",
                "walltime": 4,
                "alloc": "sup3r"
            }
        }

    Note that the ``execution_control`` block will run the job locally, while
    the ``execution_control_eagle`` block are kwargs that would be required to
    distribute the job on multiple nodes on the NREL HPC.
    """
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(fp_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def data_extract(ctx, verbose):
    """Sup3r data extraction and caching prior to training

    The sup3r data-extract module is a utility to pre-extract and pre-process
    data from a source file to disk pickle files for faster restarts while
    debugging. You can call the data-extract module via the sup3r-pipeline CLI,
    or call it directly with either of these equivelant commands::

        $ sup3r -c config_extract.json data-extract

        $ sup3r-data-extract from-config -c config_extract.json

    A sup3r data-extract config.json file can contain any arguments or keyword
    arguments required to initialize the
    :class:`sup3r.preprocessing.data_handling.DataHandler` class. The config
    also has several optional arguments: ``handler_class``, ``log_level``, and
    ``execution_control``. Here's a small example data-extract config::

        {
            "file_paths": "/datasets/WIND/conus/v1.0.0/wtk_conus_2007.h5",
            "features": ["U_100m", "V_100m"],
            "target": [27, -97],
            "shape": [800, 800],
            "execution_control": {"option": "local"},
            "log_level": "DEBUG"
        }

    Note that the ``execution_control`` has the same options as forward-pass
    and you can set ``"option": "eagle"`` to run on the NREL HPC.
    """
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(dh_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def data_collect(ctx, verbose):
    """Sup3r data collection following forward pass.

    The sup3r data-collect module can be used to collect time-chunked files
    that were spit out by multi-node forward pass jobs. You can call the
    data-collect module via the sup3r-pipeline CLI, or call it directly with
    either of these equivelant commands::

        $ sup3r -c config_collect.json data-collect

        $ sup3r-data-collect from-config -c config_collect.json

    A sup3r data-collect config.json file can contain any arguments or keyword
    arguments required to run the
    :meth:`sup3r.postprocessing.collection.Collector.collect` method. The
    config also has several optional arguments: ``log_file``, ``log_level``,
    and ``execution_control``. Here's a small example data-collect config::

        {
            "file_paths": "./outputs/*.h5",
            "out_file": "./outputs/output_file.h5",
            "features": ["windspeed_100m", "winddirection_100m"],
            "log_file": "./logs/collect.log",
            "execution_control": {"option": "local"},
            "log_level": "DEBUG"
        }

    Note that the ``execution_control`` has the same options as forward-pass
    and you can set ``"option": "eagle"`` to run on the NREL HPC.
    """
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(dc_cli, config_file=config_file, verbose=verbose)


@main.group(invoke_without_command=True)
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--monitor', is_flag=True,
              help='Flag to monitor pipeline jobs continuously. '
              'Default is not to monitor (kick off jobs and exit).')
@click.option('--background', is_flag=True,
              help='Flag to monitor pipeline jobs continuously '
              'in the background using the nohup command. Note that the '
              'stdout/stderr will not be captured, but you can set a '
              'pipeline "log_file" to capture logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def pipeline(ctx, cancel, monitor, background, verbose):
    """Execute multiple steps in a Sup3r pipeline.

    Typically, a good place to start is to set up a sup3r job with a pipeline
    config that points to several sup3r modules that you want to run in serial.
    You would call the sup3r pipeline CLI using either of these equivelant
    commands::

        $ sup3r -c config_pipeline.json pipeline

        $ sup3r-pipeline from-config -c config_pipeline.json

    A typical sup3r pipeline config.json file might look like this::

        {
            "logging": {"log_level": "DEBUG"},
            "pipeline": [
                {"forward-pass": "./config_fwp.json"},
                {"data-collect": "./config_collect.json"}
            ]
        }

    See the other CLI help pages for what the respective module configs
    require.
    """
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(pipe_cli, config_file=config_file, cancel=cancel,
                   monitor=monitor, background=background, verbose=verbose)


@pipeline.command()
@click.pass_context
def valid_pipeline_keys(ctx):
    """Print the valid pipeline config keys"""
    ctx.invoke(pipeline_keys)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r CLI')
        raise
