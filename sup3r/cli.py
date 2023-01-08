# -*- coding: utf-8 -*-
"""
Sup3r command line interface (CLI).
"""
import click
import logging

from sup3r.version import __version__
from sup3r.pipeline.forward_pass_cli import from_config as fwp_cli
from sup3r.solar.solar_cli import from_config as solar_cli
from sup3r.preprocessing.data_extract_cli import from_config as dh_cli
from sup3r.postprocessing.data_collect_cli import from_config as dc_cli
from sup3r.qa.qa_cli import from_config as qa_cli
from sup3r.qa.visual_qa_cli import from_config as visual_qa_cli
from sup3r.qa.stats_cli import from_config as stats_cli
from sup3r.pipeline.pipeline_cli import from_config as pipe_cli
from sup3r.pipeline.pipeline_cli import valid_config_keys as pipeline_keys
from sup3r.batch.batch_cli import from_config as batch_cli
from sup3r.bias.bias_calc_cli import from_config as bias_calc_cli
from sup3r.utilities.regridder_cli import from_config as regrid_cli


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
    You would call the sup3r pipeline CLI using either of these equivalent
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
    directly with either of these equivalent commands::

        $ sup3r -c config_fwp.json forward-pass

        $ sup3r-forward-pass from-config -c config_fwp.json

    A sup3r forward pass config.json file can contain any arguments or keyword
    arguments required to initialize the
    :class:`sup3r.pipeline.forward_pass.ForwardPassStrategy` module. The config
    also has several optional arguments: ``log_pattern``, ``log_level``, and
    ``execution_control``. Here's a small example forward pass config::

        {
            "file_paths": "./source_files*.nc",
            "model_kwargs": {
                "model_dir": "./sup3r_model/"
            },
            "out_pattern": "./output/sup3r_out_{file_id}.h5",
            "log_pattern": "./logs/log_{node_index}.log",
            "log_level": "DEBUG",
            "fwp_chunk_shape": [5, 5, 3],
            "spatial_pad": 1,
            "temporal_pad": 1,
            "max_nodes": 1,
            "worker_kwargs": {
                "max_workers": null,
                "output_workers": 1,
                "pass_workers": 8,
                "ti_workers": 1
            },
            "input_handler_kwargs": {
                "worker_kwargs": {
                    "max_workers": 1
                },
            },
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
    ctx.invoke(fwp_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def solar(ctx, verbose):
    """Sup3r solar module to convert GAN output clearsky ratio to irradiance

    Typically we train solar GAN's on clearsky ratio to remove the dependence
    on known variables like solar position and clearsky irradiance. This module
    converts the clearsky ratio output by the GAN in the forward-pass step to
    actual irradiance values (ghi, dni, and dhi). This should be run after the
    forward-pass but before the data-collect step.

    You can call the solar module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_solar.json solar

        $ sup3r-solar from-config -c config_solar.json

    A sup3r solar config.json file can contain any arguments or keyword
    arguments required to call the
    :meth:`sup3r.solar.solar.Solar.run_temporal_chunk` method. You do not need
    to include the ``i_t_chunk`` input, this is added by the CLI. The config
    also has several optional arguments: ``log_pattern``, ``log_level``, and
    ``execution_control``. Here's a small example solar config::

        {
            "fp_pattern": "./chunks/sup3r*.h5",
            "nsrdb_fp": "/datasets/NSRDB/current/nsrdb_2015.h5",
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
    ctx.invoke(solar_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def bias_calc(ctx, verbose):
    """Sup3r bias correction calculation module to create bias correction
    factors for low res input data.

    This is typically used to understand and correct the biases in global
    climate model (GCM) data from CMIP. Bias correcting GCM data is a very
    common process when dealing with climate data and this helps automate that
    process prior to passing GCM data through sup3r generative models. This is
    typically the first step in a GCM downscaling pipeline.

    You can call the bias calc module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_bias.json bias-calc

        $ sup3r-bias-calc from-config -c config_bias.json

    A sup3r bias calc config.json file can contain any arguments or keyword
    arguments required to call the

    The config has high level ``bias_calc_class`` and ``jobs`` keys. The
    ``bias_calc_class`` is a class name from the :mod:`sup3r.bias.bias_calc`
    module, and the ``jobs`` argument is a list of kwargs required to
    initialize the ``bias_calc_class`` and run the ``bias_calc_class.run()``
    method (for example, see
    :meth:`sup3r.bias.bias_calc.LinearCorrection.run`). There are also has
    several optional arguments: ``log_pattern``, ``log_level``, and
    ``execution_control``. Here's a small example bias calc config::

        {
            "bias_calc_class": "LinearCorrection",
            "jobs": [
                {
                    "base_fps" : ["/datasets/WIND/HRRR/HRRR_2015.h5"],
                    "bias_fps": ["./ta_day_EC-Earth3-Veg_ssp585.nc"],
                    "base_dset": "U_100m",
                    "bias_feature": "U_100m",
                    "target": [20, -130],
                    "shape": [48, 95]
                 }
            ],
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
    ctx.invoke(bias_calc_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def data_extract(ctx, verbose):
    """Sup3r data extraction and caching prior to training

    The sup3r data-extract module is a utility to pre-extract and pre-process
    data from a source file to disk pickle files for faster restarts while
    debugging. You can call the data-extract module via the sup3r-pipeline CLI,
    or call it directly with either of these equivalent commands::

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
    either of these equivalent commands::

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


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def qa(ctx, verbose):
    """Sup3r QA module following forward pass and collection.

    The sup3r QA module can be used to verify how well the high-resolution
    sup3r resolved outputs adhere to the low-resolution source data. You can
    call the QA module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_qa.json qa

        $ sup3r-qa from-config -c config_qa.json

    A sup3r QA config.json file can contain any arguments or keyword
    arguments required to initialize the :class:`sup3r.qa.qa.Sup3rQa` class.
    The config also has several optional arguments: ``log_file``,
    ``log_level``, and ``execution_control``. Here's a small example
    QA config::

        {
            "source_file_paths": "./source_files*.nc",
            "out_file_path": "./outputs/collected_output_file.h5",
            "s_enhance": 25,
            "t_enhance": 24,
            "temporal_coarsening_method": "average",
            "log_file": "./logs/qa.log",
            "execution_control": {"option": "local"},
            "log_level": "DEBUG"
        }

    Note that the ``execution_control`` has the same options as forward-pass
    and you can set ``"option": "eagle"`` to run on the NREL HPC.
    """
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(qa_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def visual_qa(ctx, verbose):
    """Sup3r visual QA module following forward pass and collection.

    The sup3r visual QA module can be used to perform a visual inspection on
    the high-resolution sup3r resolved output. You can call the visual QA
    module via the sup3r-pipeline CLI, or call it directly with either of
    these equivalent commands::

        $ sup3r -c config_qa.json visual-qa

        $ sup3r-visual-qa from-config -c config_qa.json

    A sup3r visual QA config.json file can contain any arguments or keyword
    arguments required to initialize the :class:`sup3r.qa.qa.Sup3rVisualQa`
    class. The config also has several optional arguments: ``log_file``,
    ``log_level``, and ``execution_control``. Here's a small example
    visual QA config::

        {
            "file_paths": "./outputs/collected_output*.h5",
            "out_pattern": "./outputs/plots/{feature}_{index}.png",
            "features": ['windspeed_100m', 'winddirection_100m'],
            "time_step": 100,
            "spatial_slice": [None, None, 100],
            "execution_control": {"option": "local"},
            "log_level": "DEBUG"
        }

    Note that the ``execution_control`` has the same options as forward-pass
    and you can set ``"option": "eagle"`` to run on the NREL HPC.
    """
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(visual_qa_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def stats(ctx, verbose):
    """Sup3r stats module following forward pass and collection.

    The sup3r stats module computes various statistics on wind fields of at
    given hub heights. These statistics include energy spectra, time derivative
    pdfs, velocity gradient pdfs, and vorticity pdfs.
    You can call the stats module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_stats.json stats

        $ sup3r-stats from-config -c config_stats.json

    A sup3r stats config.json file can contain any arguments or keyword
    arguments required to initialize the
    :class:`sup3r.qa.stats.Sup3rStatsMulti` class. The config also has several
    optional arguments: ``log_file``, ``log_level``, and ``execution_control``.
    Here's a small example stats config::

        {
            "source_file_paths": "./source_files*.nc",
            "out_file_path": "./outputs/collected_output_file.h5",
            "s_enhance": 2,
            "t_enhance": 12,
            "features": ["windspeed_100m", "winddirection_100m"],
            "include_stats": ["time_derivative", "gradient", "spectrum_k"]
            "get_interp": True,
            "log_file": "./logs/stats.log",
            "execution_control": {"option": "local"},
            "log_level": "DEBUG"
        }

    Note that the ``execution_control`` has the same options as forward-pass
    and you can set ``"option": "eagle"`` to run on the NREL HPC.
    """
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(stats_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def regrid(ctx, verbose):
    """Sup3r regrid module for regridding forward pass output to a different
    meta file.

    You can call the regrid module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_regrid.json regrid

        $ sup3r-regrid from-config -c config_regrid.json

    A sup3r regrid config.json file can contain any arguments or keyword
    arguments required to initialize the
    :class:`sup3r.utilities.regridder.RegridOutput` class. The config also has
    several optional arguments: ``log_file``, ``log_level``, and
    ``execution_control``.
    Here's a small example regrid config::

        {
            "source_files": "./source_files*.h5",
            "out_pattern": "./chunks_{file_id}.h5",
            "heights": [100, 200],
            "target_meta": "./target_meta.csv",
            "n_chunks": 100,
            "worker_kwargs": {"regrid_workers": 10, "query_workers": 10},
            "cache_pattern": "./{array_name}.pkl",
            "log_file": "./logs/regrid.log",
            "execution_control": {"option": "local"},
            "log_level": "DEBUG"
        }

    Note that the ``execution_control`` has the same options as forward-pass
    and you can set ``"option": "eagle"`` to run on the NREL HPC.
    """
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(regrid_cli, config_file=config_file, verbose=verbose)


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
    You would call the sup3r pipeline CLI using either of these equivalent
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


@main.group(invoke_without_command=True)
@click.option('--dry-run', is_flag=True,
              help='Flag to do a dry run (make batch dirs without running).')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given batch.')
@click.option('--delete', is_flag=True,
              help='Flag to delete all batch job sub directories associated '
              'with the batch_jobs.csv in the current batch config directory.')
@click.option('--monitor-background', is_flag=True,
              help='Flag to monitor all batch pipelines continuously '
              'in the background using the nohup command. Note that the '
              'stdout/stderr will not be captured, but you can set a '
              'pipeline "log_file" to capture logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def batch(ctx, dry_run, cancel, delete, monitor_background, verbose):
    """Create and run multiple sup3r project directories based on batch
    permutation logic.

    The sup3r batch module (built on the reV batch functionality) is a way to
    create and run many sup3r pipeline projects based on permutations of
    key-value pairs in the run config files. A user configures the batch file
    by creating one or more "sets" that contain one or more arguments (keys
    found in config files) that are to be parameterized. For example, in the
    config below, four sup3r pipelines will be created where arg1 and arg2 are
    set to [0, "a"], [0, "b"], [1, "a"], [1, "b"] in config_fwp.json::

        {
            "pipeline_config": "./config_pipeline.json",
            "sets": [
              {
                "args": {
                  "arg1": [0, 1],
                  "arg2": ["a", "b"],
                },
                "files": ["./config_fwp.json"],
                "set_tag": "set1"
              }
        }

    Note that you can use multiple "sets" to isolate parameter permutations.
    """
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(batch_cli, config_file=config_file,
                   dry_run=dry_run, cancel=cancel, delete=delete,
                   monitor_background=monitor_background,
                   verbose=verbose)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r CLI')
        raise
