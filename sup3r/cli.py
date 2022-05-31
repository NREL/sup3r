# -*- coding: utf-8 -*-
"""
sup3r command line interface (CLI).
"""
import click
import logging

from sup3r.version import __version__
from sup3r.pipeline.forward_pass_cli import from_config as fp_cli
from sup3r.preprocessing.data_extract_cli import from_config as dh_cli


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


@main.group()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def forward_pass(ctx, verbose):
    """sup3r forward pass to super-resolve data."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(fp_cli, config_file=config_file, verbose=verbose)


@main.group()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging.')
@click.pass_context
def data_extract(ctx, verbose):
    """sup3r data extraction and caching prior to training or forward pass."""
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any([verbose, ctx.obj['VERBOSE']])
    ctx.invoke(dh_cli, config_file=config_file, verbose=verbose)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r CLI')
        raise
