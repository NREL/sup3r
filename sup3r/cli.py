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
            Dictionary of kwargs from the nsrdb config file specifically for
            this command block.
        eagle_args : dict
            Dictionary of kwargs from the nsrdb config to make eagle submission
        """
        file_path = cmd_args['file_path']
        temporal_res = cmd_args['temporal_res']
        spatial_res = cmd_args['spatial_res']
        target_lat = cmd_args['target_lat']
        target_lon = cmd_args['target_lon']
        shape = cmd_args['shape']
        features = cmd_args['features']
        n_observations = cmd_args['n_observations']
        ctx.invoke(data_model, file_path=file_path,
                   n_observations=n_observations,
                   temporal_res=temporal_res,
                   spatial_res=spatial_res,
                   target=(target_lat, target_lon),
                   shape=(shape, shape),
                   features=features)
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
@click.option('--doy', '-d', type=int, required=True,
              help='Integer day-of-year to run data model for.')
@click.option('--var_list', '-vl', type=STRLIST, required=False, default=None,
              help='Variables to process with the data model. None will '
              'default to all NSRDB variables.')
@click.option('--dist_lim', '-dl', type=FLOAT, required=True, default=1.0,
              help='Return only neighbors within this distance during cloud '
              'regrid. The distance is in decimal degrees (more efficient '
              'than real distance). NSRDB sites further than this value from '
              'GOES data pixels will be warned and given missing cloud types '
              'and properties resulting in a full clearsky timeseries.')
@click.option('--factory_kwargs', '-kw', type=DICT,
              required=False, default=None,
              help='Optional namespace of kwargs to use to initialize '
              'variable data handlers from the data models variable factory. '
              'Keyed by variable name. Values can be "source_dir", "handler", '
              'etc... source_dir for cloud variables can be a normal '
              'directory path or /directory/prefix*suffix where /directory/ '
              'can have more sub dirs.')
@click.option('--max_workers', '-w', type=INT, default=None,
              help='Number of workers to use in parallel.')
@click.option('--max_workers_regrid', '-mwr', type=INT, default=None,
              help='Number of workers to use in parallel for the '
                   'cloud regrid algorithm.')
@click.option('--max_workers_cloud_io', '-mwc', type=INT, default=None,
              help='Number of workers to use in parallel for the '
                   'cloud data io.')
@click.option('-ml', '--mlclouds', is_flag=True,
              help='Flag to process additional variables if mlclouds gap fill'
              'is going to be run after the data_model step.')
@click.pass_context
def data_model(ctx, file_path, n_observations,
               temporal_res, spatial_res,
               target, shape, features,
               factory_kwargs):
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
    fun_str = 'preprocessing.run_data_model'
    arg_str = (f'h5_path={file_path}, '
               f'n_observations={n_observations}, '
               f'temporal_res={temporal_res}, '
               f'spatial_res={spatial_res}, '
               f'target={target}, '
               f'shape={shape}, '
               f'features={features}, ')

    ctx.obj['IMPORT_STR'] = 'from sup3r.data_model import preprocessing '
    ctx.obj['FUN_STR'] = fun_str
    ctx.obj['ARG_STR'] = arg_str
    ctx.obj['COMMAND'] = 'data-model'


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

    msg = 'NSRDB CLI failed to submit jobs!'
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
