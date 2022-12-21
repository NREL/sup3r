# -*- coding: utf-8 -*-
"""
sup3r data extraction CLI entry points.
"""
import click
import logging
import os

from rex.utilities.loggers import init_mult

import sup3r
from sup3r.pipeline.config import BaseConfig
from sup3r.utilities import ModuleName
from sup3r.version import __version__
from sup3r.cli.base import BaseCLI


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """Sup3r Data Extraction Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='sup3r data extract configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run sup3r data extraction from a config file.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    config_file : str
        Filepath to sup3r data extraction json file.
    verbose : bool
        Flag to turn on debug logging. Default is not verbose.
    """
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    status_dir = os.path.dirname(os.path.abspath(config_file))
    ctx.obj['OUT_DIR'] = status_dir
    config = BaseConfig(config_file)
    config_verbose = config.get('log_level', 'INFO')
    config_verbose = (config_verbose == 'DEBUG')
    config_handler = config.get('handler_class', 'DataHandler')
    verbose = any([verbose, config_verbose, ctx.obj['VERBOSE']])

    init_mult('sup3r_data_extract', os.path.join(status_dir, 'logs/'),
              modules=[__name__, 'sup3r'], verbose=verbose)

    exec_kwargs = config.get('execution_control', {})
    exec_kwargs['stdout_path'] = os.path.join(status_dir, 'stdout/')
    hardware_option = exec_kwargs.pop('option', 'local')
    logger.debug('Found execution kwargs: {}'.format(exec_kwargs))
    logger.debug('Hardware run option: "{}"'.format(hardware_option))

    HANDLER_CLASS = getattr(sup3r.preprocessing.data_handling, config_handler)

    name = 'sup3r_extract_{}'.format(os.path.basename(status_dir))
    ctx.obj['NAME'] = name
    config['job_name'] = name
    config['status_dir'] = status_dir

    cmd = HANDLER_CLASS.get_node_cmd(config)
    logger.debug(f'Running command: {cmd}')

    if hardware_option.lower() in ('eagle', 'slurm'):
        kickoff_slurm_job(ctx, cmd, **exec_kwargs)
    else:
        kickoff_local_job(ctx, cmd)


def kickoff_local_job(ctx, cmd):
    """Run sup3r data extraction locally.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in shell script. Example:
            'python -m sup3r.cli data_extract -c <config_file>'
    """
    BaseCLI.kickoff_local_job(ModuleName.DATA_EXTRACT, ctx, cmd)


def kickoff_slurm_job(ctx, cmd, alloc='sup3r', memory=None, walltime=4,
                      feature=None, stdout_path='./stdout/'):
    """Run sup3r on HPC via SLURM job submission.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in SLURM shell script. Example:
            'python -m sup3r.cli data_extract -c <config_file>'
    alloc : str
        HPC project (allocation) handle. Example: 'sup3r'.
    memory : int
        Node memory request in GB.
    walltime : float
        Node walltime request in hours.
    feature : str
        Additional flags for SLURM job. Format is "--qos=high"
        or "--depend=[state:job_id]". Default is None.
    stdout_path : str
        Path to print .stdout and .stderr files.
    """
    BaseCLI.kickoff_slurm_job(ModuleName.DATA_EXTRACT, ctx, cmd, alloc, memory,
                              walltime, feature, stdout_path)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r data extraction CLI')
        raise
