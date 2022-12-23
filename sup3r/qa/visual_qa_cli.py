# -*- coding: utf-8 -*-
"""
sup3r visual QA module CLI entry points.
"""
import click
import logging

from sup3r.utilities import ModuleName
from sup3r.version import __version__
from sup3r.qa.visual_qa import Sup3rVisualQa
from sup3r.utilities.cli import BaseCLI

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """Sup3r visual QA module Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='sup3r visual QA configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run the sup3r visual QA module from a config file."""
    BaseCLI.from_config(ModuleName.VISUAL_QA, Sup3rVisualQa, ctx, config_file,
                        verbose)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r visual QA CLI')
        raise
