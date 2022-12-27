# -*- coding: utf-8 -*-
"""
sup3r solar CLI entry points.
"""
import copy
import click
import logging
import os

from sup3r.solar import Solar
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
    """Sup3r Solar Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='sup3r solar configuration .json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run sup3r solar from a config file."""
    config = BaseCLI.from_config_preflight(ModuleName.SOLAR, ctx, config_file,
                                           verbose)
    exec_kwargs = config.get('execution_control', {})
    hardware_option = exec_kwargs.pop('option', 'local')
    log_pattern = config['log_pattern']
    fp_pattern = config['fp_pattern']
    basename = config['job_name']
    fp_sets, _, temporal_ids, _, _ = Solar.get_sup3r_fps(fp_pattern)
    logger.info('Solar module found {} sets of chunked source files to run '
                'on. Submitting to {} nodes based on the number of temporal '
                'chunks'.format(len(fp_sets), len(set(temporal_ids))))

    for i_node, temporal_id in enumerate(sorted(set(temporal_ids))):
        node_config = copy.deepcopy(config)
        node_config['log_file'] = (
            log_pattern if log_pattern is None
            else os.path.normpath(log_pattern.format(node_index=i_node)))
        name = ('{}_{}'.format(basename, str(i_node).zfill(6)))
        ctx.obj['NAME'] = name
        node_config['job_name'] = name

        node_config['temporal_id'] = temporal_id
        cmd = Solar.get_node_cmd(node_config)

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
    BaseCLI.kickoff_slurm_job(ModuleName.SOLAR, ctx, cmd, alloc, memory,
                              walltime, feature, stdout_path)


def kickoff_local_job(ctx, cmd):
    """Run sup3r solar locally.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in shell script. Example:
            'python -m sup3r.cli forward_pass -c <config_file>'
    """
    BaseCLI.kickoff_local_job(ModuleName.SOLAR, ctx, cmd)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r solar CLI')
        raise
