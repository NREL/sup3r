# -*- coding: utf-8 -*-
"""
sup3r forward pass CLI entry points.
"""
import os
import json
import click
import logging

from reV.pipeline.stats import Status

from rex.utilities.execution import SubprocessManager
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_mult
from rex.utilities.cli_dtypes import STR

from sup3r.version import __version__
from sup3r.pipeline.forward_pass import ForwardPassStrategy, ForwardPass


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('--name', '-n', default=os.path.basename(os.getcwd()),
              type=STR, show_default=True, help='sup3r forward pass job name.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, name, verbose):
    """reV Generation Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['NAME'] = name
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='sup3r forward pass configuration json file.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, verbose):
    """Run sup3r forward pass from a config file.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    config_file : str
        Filepath to sup3r forward pass json file.
    verbose : bool
        Flag to turn on debug logging. Default is not verbose.
    """
    verbose = any([verbose, ctx.obj['VERBOSE']])

    with open(config_file, 'r') as f:
        config = json.load(f)

    name = ctx.obj['NAME']

    init_mult(name, './logs/', modules=[__name__, 'sup3r'], verbose=verbose)

    # PSEUDOCODE
    strategy = ForwardPassStrategy(config)
    for i, node in enumerate(strategy):
        name = 'sup3r_fwp_{}'.format(i)
        ctx.obj['NAME'] = name
        config.update(node.kwargs)
        cmd = ForwardPass.get_node_cmd(config)

        if config.slurm:
            kickoff_slurm_job(ctx, cmd, slurm_kwargs)
        else:
            SubprocessManager.submit(cmd)


def kickoff_slurm_job(ctx, cmd, alloc, memory=None, walltime=4, feature=None,
                      stdout_path='./stdout/'):
    """Run sup3r on HPC via SLURM job submission.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in SLURM shell script. Example:
            'python -m reV.generation.cli_gen'
    alloc : str
        HPC project (allocation) handle. Example: 'rev'. Default is not to
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
    fn_out = ctx.obj['FN_OUT']

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

        # add job to reV status file.
        Status.add_job(out_dir, module=ModuleName.FORWARD_PASS,
                       job_name=name, replace=True,
                       job_attrs={'job_id': out, 'hardware': 'slurm',
                                  'fn_out': fn_out, 'out_dir': out_dir})

        click.echo(msg)
        logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r forward pass CLI')
        raise
