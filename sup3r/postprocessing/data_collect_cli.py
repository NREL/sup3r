# -*- coding: utf-8 -*-
"""
sup3r data collection CLI entry points.
"""
import click
import logging
import copy

from sup3r.utilities import ModuleName
from sup3r.version import __version__
from sup3r.postprocessing.collection import Collector
from sup3r.utilities.cli import BaseCLI


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """Sup3r Data Collection Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='sup3r data collection configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run sup3r data collection from a config file. If dset_split is True this
    each feature will be collected into a separate file."""
    config = BaseCLI.from_config_preflight(ModuleName.DATA_COLLECT, ctx,
                                           config_file, verbose)

    dset_split = config.get('dset_split', False)
    exec_kwargs = config.get('execution_control', {})
    hardware_option = exec_kwargs.pop('option', 'local')

    configs = [config]
    if dset_split:
        configs = []
        for feature in config['features']:
            f_config = copy.deepcopy(config)
            f_out_file = config['out_file'].replace('.h5', f'_{feature}.h5')
            f_job_name = config['job_name'] + f'_{feature}'
            f_log_file = config.get('log_file', None)
            if f_log_file is not None:
                f_log_file = f_log_file.replace('.log', f'_{feature}.log')
            f_config.update({'features': [feature],
                             'out_file': f_out_file,
                             'job_name': f_job_name,
                             'log_file': f_log_file})

            configs.append(f_config)

    for config in configs:
        ctx.obj['NAME'] = config['job_name']
        cmd = Collector.get_node_cmd(config)

        cmd_log = '\n\t'.join(cmd.split('\n'))
        logger.debug(f'Running command:\n\t{cmd_log}')

        if hardware_option.lower() in ('eagle', 'slurm'):
            kickoff_slurm_job(ctx, cmd, **exec_kwargs)
        else:
            kickoff_local_job(ctx, cmd)


def kickoff_local_job(ctx, cmd):
    """Run sup3r data collection locally.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in shell script. Example:
            'python -m sup3r.cli data_collect -c <config_file>'
    """
    BaseCLI.kickoff_local_job(ModuleName.DATA_COLLECT, ctx, cmd)


def kickoff_slurm_job(ctx, cmd, alloc='sup3r', memory=None, walltime=4,
                      feature=None, stdout_path='./stdout/'):
    """Run sup3r on HPC via SLURM job submission.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in SLURM shell script. Example:
            'python -m sup3r.cli data-collect -c <config_file>'
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
    BaseCLI.kickoff_slurm_job(ModuleName.DATA_COLLECT, ctx, cmd, alloc, memory,
                              walltime, feature, stdout_path)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r data collection CLI')
        raise
