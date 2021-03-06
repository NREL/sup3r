# -*- coding: utf-8 -*-
"""
sup3r forward pass CLI entry points.
"""
import copy
import click
import logging
from inspect import signature
import os

from reV.pipeline.status import Status

from rex.utilities.execution import SubprocessManager
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_mult
from rex.utilities.utilities import safe_json_load

from sup3r.utilities import ModuleName
from sup3r.version import __version__
from sup3r.pipeline.forward_pass import ForwardPassStrategy
from sup3r.pipeline.forward_pass import ForwardPass


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """Sup3r Forward Pass Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='sup3r forward pass configuration .json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run sup3r forward pass from a config file."""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    status_dir = os.path.dirname(os.path.abspath(config_file))
    ctx.obj['OUT_DIR'] = status_dir
    config = safe_json_load(config_file)
    config['status_dir'] = status_dir
    config_verbose = config.get('log_level', 'INFO')
    config_verbose = (config_verbose == 'DEBUG')
    verbose = any([verbose, config_verbose, ctx.obj['VERBOSE']])

    init_mult('sup3r_fwp', './logs/', modules=[__name__, 'sup3r'],
              verbose=verbose)

    exec_kwargs = config.get('execution_control', {})
    log_pattern = config.get('log_pattern', None)
    if log_pattern is not None:
        if '.log' not in log_pattern:
            log_pattern += '.log'
        if '{node_index}' not in log_pattern:
            log_pattern = log_pattern.replace('.log', '_{node_index}.log')

    logger.debug('Found execution kwargs: {}'.format(exec_kwargs))
    hardware_option = exec_kwargs.pop('option', 'local')
    logger.debug('Hardware run option: "{}"'.format(hardware_option))

    sig = signature(ForwardPassStrategy)
    strategy_kwargs = {k: v for k, v in config.items()
                       if k in sig.parameters.keys()}
    strategy = ForwardPassStrategy(**strategy_kwargs)

    node_index = config.get('node_index', None)
    nodes = [node_index] if node_index is not None else range(strategy.nodes)
    for i in nodes:
        node_config = copy.deepcopy(config)
        node_config['node_index'] = i
        node_config['log_file'] = (
            log_pattern if log_pattern is None
            else os.path.normpath(log_pattern.format(node_index=i)))
        name = 'sup3r_fwp_{}'.format(str(i).zfill(5))
        ctx.obj['NAME'] = name
        node_config['job_name'] = name
        cmd = ForwardPass.get_node_cmd(node_config)

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
        HPC project (allocation) handle. Example: 'sup3r'. Default is not to
        state an allocation (does not work on Eagle slurm).
    memory : int
        Node memory request in GB.
    walltime : float
        Node walltime request in hours. Default is not to state a walltime
        (does not work on Eagle slurm).
    feature : str
        Additional flags for SLURM job. Format is "--qos=high"
        or "--depend=[state:job_id]". Default is None.
    stdout_path : str
        Path to print .stdout and .stderr files.
    """

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
    if slurm_manager is None:
        slurm_manager = SLURM()
        ctx.obj['SLURM_MANAGER'] = slurm_manager

    status = Status.retrieve_job_status(out_dir,
                                        module=ModuleName.FORWARD_PASS,
                                        job_name=name,
                                        hardware='slurm',
                                        subprocess_manager=slurm_manager)

    msg = 'sup3r forward pass CLI failed to submit jobs!'
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'.format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               .format(name, status))
    else:
        logger.info('Running sup3r forward pass on SLURM with node name "{}".'
                    .format(name))
        out = slurm_manager.sbatch(cmd,
                                   alloc=alloc,
                                   memory=memory,
                                   walltime=walltime,
                                   feature=feature,
                                   name=name,
                                   stdout_path=stdout_path)[0]
        if out:
            msg = ('Kicked off sup3r forward pass job "{}" (SLURM jobid #{}).'
                   .format(name, out))

        # add job to sup3r status file.
        Status.add_job(out_dir, module=ModuleName.FORWARD_PASS,
                       job_name=name, replace=True,
                       job_attrs={'job_id': out, 'hardware': 'slurm'})

    click.echo(msg)
    logger.info(msg)


def kickoff_local_job(ctx, cmd):
    """Run sup3r forward pass locally.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in shell script. Example:
            'python -m sup3r.cli forward_pass -c <config_file>'
    """

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    subprocess_manager = SubprocessManager
    status = Status.retrieve_job_status(out_dir,
                                        module=ModuleName.FORWARD_PASS,
                                        job_name=name)
    msg = 'sup3r forward pass CLI failed to submit jobs!'
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'.format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               .format(name, status))
    else:
        logger.info('Running sup3r forward pass locally with job name "{}".'
                    .format(name))
        Status.add_job(out_dir, module=ModuleName.FORWARD_PASS,
                       job_name=name, replace=True)
        subprocess_manager.submit(cmd)
        msg = ('Completed sup3r forward pass job "{}".'.format(name))

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r forward pass CLI')
        raise
