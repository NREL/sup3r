# -*- coding: utf-8 -*-
"""
sup3r bias correction calculation CLI entry points.
"""
import copy
import click
import logging
import os

import sup3r.bias.bias_calc
from sup3r.utilities import ModuleName
from sup3r.version import __version__
from sup3r.utilities.cli import BaseCLI


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """Sup3r bias calc Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='sup3r bias correction calculation config .json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run sup3r bias correction calculation from a config file."""
    config = BaseCLI.from_config_preflight(ModuleName.BIAS_CALC, ctx,
                                           config_file, verbose)

    exec_kwargs = config.get('execution_control', {})
    hardware_option = exec_kwargs.pop('option', 'local')
    calc_class_name = config['bias_calc_class']
    BiasCalcClass = getattr(sup3r.bias.bias_calc, calc_class_name)
    basename = config['job_name']
    log_pattern = config.get('log_pattern', None)

    jobs = config['jobs']
    for i_node, job in enumerate(jobs):
        node_config = copy.deepcopy(job)
        node_config['status_dir'] = config['status_dir']
        node_config['log_file'] = (
            log_pattern if log_pattern is None
            else os.path.normpath(log_pattern.format(node_index=i_node)))
        name = ('{}_{}'.format(basename, str(i_node).zfill(6)))
        ctx.obj['NAME'] = name
        node_config['job_name'] = name

        cmd = BiasCalcClass.get_node_cmd(node_config)

        cmd_log = '\n\t'.join(cmd.split('\n'))
        logger.debug(f'Running command:\n\t{cmd_log}')

        if hardware_option.lower() in ('eagle', 'slurm'):
            kickoff_slurm_job(ctx, cmd, **exec_kwargs)
        else:
            kickoff_local_job(ctx, cmd)


def kickoff_slurm_job(ctx, cmd, alloc='sup3r', memory=None, walltime=4,
                      feature=None, stdout_path='./stdout/'):
    """Run sup3r on HPC via SLURM job submission.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in SLURM shell script. Example:
            'python -m sup3r.cli forward_pass -c <config_file>'
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
    BaseCLI.kickoff_slurm_job(ModuleName.BIAS_CALC, ctx, cmd, alloc, memory,
                              walltime, feature, stdout_path)


def kickoff_local_job(ctx, cmd):
    """Run sup3r bias calc locally.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in shell script. Example:
            'python -m sup3r.cli forward_pass -c <config_file>'
    """
    BaseCLI.kickoff_local_job(ModuleName.BIAS_CALC, ctx, cmd)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r bias calc CLI')
        raise
