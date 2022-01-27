# -*- coding: utf-8 -*-
"""Sup3r Command Line Interface (CLI).

Created on January 26 2022

@author: bnb32
"""
import click
import json
import logging
import os

from rex.utilities.cli_dtypes import STR, INT, FLOAT, STRLIST
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_logger
from rex.utilities.utilities import safe_json_load

from sup3r.pipeline import Status, Sup3rPipeline

logger = logging.getLogger(__name__)


class DictType(click.ParamType):
    """Dict click input argument type."""

    name = 'dict'

    @staticmethod
    def convert(value, param, ctx):
        """Convert to dict or return as None."""
        if isinstance(value, dict):
            return value
        elif isinstance(value, str):
            return json.loads(value)
        elif value is None:
            return None
        else:
            raise TypeError('Cannot recognize int type: {} {} {} {}'
                            .format(value, type(value), param, ctx))


DICT = DictType()


@click.group()
@click.pass_context
def main(ctx):
    """Sup3r processing CLI."""
    ctx.ensure_object(dict)


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='Sup3r pipeline configuration json file.')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--monitor', is_flag=True,
              help='Flag to monitor pipeline jobs continuously. '
              'Default is not to monitor (kick off jobs and exit).')
@click.pass_context
def pipeline(ctx, config_file, cancel, monitor):
    """Sup3r pipeline from a pipeline config file."""

    ctx.ensure_object(dict)
    if cancel:
        Sup3rPipeline.cancel_all(config_file)
    else:
        Sup3rPipeline.run(config_file, monitor=monitor)


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='Filepath to config file.')
@click.option('--command', '-cmd', type=str, required=True,
              help='Sup3r CLI command string.')
@click.pass_context
def config(ctx, config_file, command):
    """Sup3r processing CLI from config json file."""

    run_config = safe_json_load(config_file)

    direct_args = run_config.pop('direct')
    eagle_args = run_config.pop('eagle')
    cmd_args = run_config.pop(command)

    if cmd_args is None:
        cmd_args = {}

    # replace any args with higher priority entries in command dict
    for k in eagle_args.keys():
        if k in cmd_args:
            eagle_args[k] = cmd_args[k]
    for k in direct_args.keys():
        if k in cmd_args:
            direct_args[k] = cmd_args[k]

    name = direct_args['name']
    ctx.obj['NAME'] = name
    ctx.obj['YEAR'] = direct_args['year']
    ctx.obj['OUT_DIR'] = direct_args['out_dir']
    ctx.obj['LOG_LEVEL'] = direct_args['log_level']
    ctx.obj['SLURM_MANAGER'] = SLURM()

    init_logger('sup3r.cli', log_level=direct_args['log_level'], log_file=None)

    if command == 'data-model':
        ConfigRunners.run_data_model_config(ctx, name, cmd_args, eagle_args)
    else:
        raise KeyError('Command not recognized: "{}"'.format(command))


class ConfigRunners:
    """Class to hold static methods that kickoff sup3r modules from extracted
    sup3r config objects"""

    @staticmethod
    def run_data_model_config(ctx, name, cmd_args, eagle_args):
        """Run the data model processing code.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object.
        name : str
            Base jobname.
        cmd_args : dict
            Dictionary of kwargs from the sup3r config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the sup3r config to make eagle submission
        """

        factory_kwargs = cmd_args.get('factory_kwargs', None)
        var_kwargs = cmd_args['var_kwargs']

        ctx.obj['NAME'] = name
        ctx.invoke(data_model, var_kwargs=var_kwargs,
                   factory_kwargs=factory_kwargs)
        ctx.invoke(eagle, **eagle_args)


@main.group()
@click.option('--name', '-n', default='Sup3r', type=str,
              help='Job and node name.')
@click.option('--year', '-y', default=None, type=INT,
              help='Year of analysis.')
@click.option('--out_dir', '-od', type=STR, required=True,
              help='Output directory.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def direct(ctx, name, year, out_dir, verbose):
    """Sup3r direct processing CLI (no config file)."""

    ctx.obj['NAME'] = name
    ctx.obj['YEAR'] = year
    ctx.obj['OUT_DIR'] = out_dir

    if verbose:
        ctx.obj['LOG_LEVEL'] = 'DEBUG'
    else:
        ctx.obj['LOG_LEVEL'] = 'INFO'


@direct.group()
@click.option('--factory_kwargs', '-kw', type=DICT,
              required=False, default=None,
              help='Optional namespace of kwargs')
@click.pass_context
def data_model(ctx, var_kwargs, factory_kwargs):
    """Run the preprocessing routine"""

    name = ctx.obj['NAME']
    year = ctx.obj['YEAR']
    out_dir = ctx.obj['OUT_DIR']
    log_level = ctx.obj['LOG_LEVEL']

    if factory_kwargs is not None:
        factory_kwargs = json.dumps(factory_kwargs)
        factory_kwargs = factory_kwargs.replace('true', 'True')
        factory_kwargs = factory_kwargs.replace('false', 'False')
        factory_kwargs = factory_kwargs.replace('null', 'None')

    log_file = 'data_model/data_model.log'
    fun_str = 'data_model.run_data_model'
    arg_str = (f'factory_kwargs={factory_kwargs}, '
               f'var_kwargs={json.dumps(var_kwargs)}, '
               f'year={year}, '
               f'job_name="{name}", '
               f'log_file="{log_file}", '
               f'out_dir="{out_dir}", '
               f'log_level="{log_level}" ')

    ctx.obj['IMPORT_STR'] = 'from sup3r.data_model import data_model '
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'data-model'


@data_model.command()
@click.option('--alloc', '-a', required=True, type=STR,
              help='Eagle allocation account name.')
@click.option('--memory', '-mem', default=None, type=INT,
              help='Eagle node memory request in GB. Default is None')
@click.option('--walltime', '-wt', default=1.0, type=float,
              help='Eagle walltime request in hours. Default is 1.0')
@click.option('--feature', '-l', default=None, type=STR,
              help=('Additional flags for SLURM job. Format is "--qos=high" '
                    'or "--depend=[state:job_id]". Default is None.'))
@click.option('--stdout_path', '-sout', default=None, type=STR,
              help='Subprocess standard output path. Default is in out_dir.')
@click.pass_context
def eagle(ctx, alloc, memory, walltime, feature, stdout_path):
    """Eagle submission tool for the Sup3r cli."""

    name = ctx.obj['NAME']
    out_dir = ctx.obj['OUT_DIR']
    import_str = ctx.obj['IMPORT_STR']
    fun_str = ctx.obj['FUN_STR']
    arg_str = ctx.obj['ARG_STR']
    command = ctx.obj['COMMAND']

    if 'SLURM_MANAGER' not in ctx.obj:
        ctx.obj['SLURM_MANAGER'] = SLURM()

    slurm_manager = ctx.obj['SLURM_MANAGER']

    if stdout_path is None:
        stdout_path = os.path.join(out_dir, 'stdout/')

    status = Status.retrieve_job_status(out_dir, command, name,
                                        hardware='eagle',
                                        subprocess_manager=slurm_manager)

    msg = 'Sup3r CLI failed to submit jobs!'
    if status == 'successful':
        msg = ('Job "{}" is successful in status json found in "{}", '
               'not re-running.'.format(name, out_dir))
    elif 'fail' not in str(status).lower() and status is not None:
        msg = ('Job "{}" was found with status "{}", not resubmitting'
               'not re-running.'.format(name, status))
    else:
        cmd = ("python -c '{import_str};{f}({a})'"
               .format(import_str=import_str, f=fun_str, a=arg_str))
        slurm_id = None
        out = slurm_manager.sbatch(cmd,
                                   alloc=alloc,
                                   memory=memory,
                                   walltime=walltime,
                                   feature=feature,
                                   name=name,
                                   stdout_path=stdout_path)[0]

        if out:
            slurm_id = out
            msg = ('Kicked off job "{}" (SLURM jobid #{}) on Eagle.'
                   .format(name, slurm_id))

        Status.add_job(
            out_dir, command, name, replace=True,
            job_attrs={'job_id': slurm_id,
                       'hardware': 'eagle',
                       'out_dir': out_dir})

    click.echo(msg)
    logger.info(msg)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.error('Error running Sup3r CLI.')
        raise
