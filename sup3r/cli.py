# -*- coding: utf-8 -*-
"""
sup3r command line interface (CLI).
"""
from operator import inv
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
    """sup3r command line interface."""
    ctx.ensure_object(dict)
    ctx.obj['CONFIG_FILE'] = config_file
    ctx.obj['VERBOSE'] = verbose


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def forward_pass(ctx, verbose):
    """sup3r forward pass to super-resolve data."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(fp_cli, config_file=config_file, verbose=verbose)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def data_extract(ctx, verbose):
    """sup3r data extraction and caching prior to training or forward pass."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(dh_cli, config_file=config_file, verbose=verbose)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def data_collect(ctx, verbose):
    """sup3r data collection following forward pass."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(dc_cli, config_file=config_file, verbose=verbose)


@main.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def data_collect(ctx, verbose):
    """sup3r data collection following forward pass."""
    if ctx.invoked_subcommand is None:
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
    """Execute multiple steps in a sup3r pipeline."""
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any([verbose, ctx.obj['VERBOSE']])
        ctx.invoke(pipe_cli, config_file=config_file, cancel=cancel,
                   monitor=monitor, background=background, verbose=verbose)


@pipeline.command()
@click.pass_context
def valid_pipeline_keys(ctx):
    """
    Valid Pipeline config keys
    """
    ctx.invoke(pipeline_keys)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r CLI')
        raise
