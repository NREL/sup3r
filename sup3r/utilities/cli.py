
# -*- coding: utf-8 -*-
"""
Sup3r base CLI class.
"""
import click
import logging
import os
import json

from reV.pipeline.status import Status

from rex.utilities.execution import SubprocessManager
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_mult

from sup3r.version import __version__
from sup3r.pipeline.config import BaseConfig
from sup3r.utilities import ModuleName


logger = logging.getLogger(__name__)


class BaseCLI:
    """Base CLI class used to create CLI for modules in ModuleName"""

    @classmethod
    def from_config(cls, module_name, module_class, ctx, config_file, verbose):
        """Run sup3r module from a config file.


        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        module_class : Object
            Class object used to call get_node_cmd(config).
            e.g. Sup3rQa.get_node_cmd(config)
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        config_file : str
            Path to config file provided all needed inputs to module_class
        verbose : bool
            Whether to run in verbose mode.
        """
        config = cls.from_config_preflight(module_name, ctx, config_file,
                                           verbose)

        exec_kwargs = config.get('execution_control', {})
        hardware_option = exec_kwargs.pop('option', 'local')

        cmd = module_class.get_node_cmd(config)

        cmd_log = '\n\t'.join(cmd.split('\n'))
        logger.debug(f'Running command:\n\t{cmd_log}')

        if hardware_option.lower() in ('eagle', 'slurm'):
            cls.kickoff_slurm_job(module_name, ctx, cmd, **exec_kwargs)
        else:
            cls.kickoff_local_job(module_name, ctx, cmd)

    @classmethod
    def from_config_preflight(cls, module_name, ctx, config_file, verbose):
        """Parse conifg file prior to running sup3r module.

        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        module_class : Object
            Class object used to call get_node_cmd(config).
            e.g. Sup3rQa.get_node_cmd(config)
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        config_file : str
            Path to config file provided all needed inputs to module_class
        verbose : bool
            Whether to run in verbose mode.

        Returns
        -------
        config : dict
            Dictionary corresponding to config_file
        """
        cls.check_module_name(module_name)

        ctx.ensure_object(dict)
        ctx.obj['VERBOSE'] = verbose
        status_dir = os.path.dirname(os.path.abspath(config_file))
        ctx.obj['OUT_DIR'] = status_dir
        config = BaseConfig(config_file)
        config['status_dir'] = status_dir
        log_file = config.get('log_file', None)
        log_pattern = config.get('log_pattern', None)
        config_verbose = config.get('log_level', 'INFO')
        config_verbose = (config_verbose == 'DEBUG')
        verbose = any([verbose, config_verbose, ctx.obj['VERBOSE']])
        exec_kwargs = config.get('execution_control', {})
        hardware_option = exec_kwargs.get('option', 'local')

        log_dir = log_file or log_pattern
        log_dir = log_dir if log_dir is None else os.path.dirname(log_dir)

        init_mult(f'sup3r_{module_name.replace("-", "_")}',
                  log_dir, modules=[__name__, 'sup3r'], verbose=verbose)

        if log_pattern is not None:
            os.makedirs(os.path.dirname(log_pattern), exist_ok=True)
            if '.log' not in log_pattern:
                log_pattern += '.log'
            if '{node_index}' not in log_pattern:
                log_pattern = log_pattern.replace('.log', '_{node_index}.log')

        exec_kwargs['stdout_path'] = os.path.join(status_dir, 'stdout/')
        logger.debug('Found execution kwargs: {}'.format(exec_kwargs))
        logger.debug('Hardware run option: "{}"'.format(hardware_option))

        name = f'sup3r_{module_name.replace("-", "_")}'
        name += '_{}'.format(os.path.basename(status_dir))
        job_name = config.get('job_name', None)
        if job_name is not None:
            name = job_name
        ctx.obj['NAME'] = name
        config['job_name'] = name
        config['status_dir'] = status_dir

        return config

    @classmethod
    def check_module_name(cls, module_name):
        """Make sure module_name is a valid member of the ModuleName class"""
        msg = ('Module name must be in ModuleName class. Received '
               f'{module_name}.')
        assert module_name in ModuleName, msg

    @classmethod
    def kickoff_slurm_job(cls, module_name, ctx, cmd, alloc='sup3r',
                          memory=None, walltime=4, feature=None,
                          stdout_path='./stdout/'):
        """Run sup3r module on HPC via SLURM job submission.

        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        cmd : str
            Command to be submitted in SLURM shell script. Example:
                'python -m sup3r.cli <module_name> -c <config_file>'
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
        cls.check_module_name(module_name)

        name = ctx.obj['NAME']
        out_dir = ctx.obj['OUT_DIR']
        slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
        if slurm_manager is None:
            slurm_manager = SLURM()
            ctx.obj['SLURM_MANAGER'] = slurm_manager

        status = Status.retrieve_job_status(out_dir,
                                            module=module_name,
                                            job_name=name,
                                            hardware='slurm',
                                            subprocess_manager=slurm_manager)

        msg = f'sup3r {module_name} CLI failed to submit jobs!'
        if status == 'successful':
            msg = (f'Job "{name}" is successful in status json found in '
                   f'"{out_dir}", not re-running.')
        elif 'fail' not in str(status).lower() and status is not None:
            msg = (f'Job "{name}" was found with status "{status}", not '
                   'resubmitting')
        else:
            logger.info(f'Running sup3r {module_name} on SLURM with node '
                        f'name "{name}".')
            out = slurm_manager.sbatch(cmd,
                                       alloc=alloc,
                                       memory=memory,
                                       walltime=walltime,
                                       feature=feature,
                                       name=name,
                                       stdout_path=stdout_path)[0]
            if out:
                msg = (f'Kicked off sup3r {module_name} job "{name}" '
                       f'(SLURM jobid #{out}).')

            # add job to sup3r status file.
            Status.add_job(out_dir, module=module_name,
                           job_name=name, replace=True,
                           job_attrs={'job_id': out, 'hardware': 'slurm'})

        click.echo(msg)
        logger.info(msg)

    @classmethod
    def kickoff_local_job(cls, module_name, ctx, cmd):
        """Run sup3r module locally.

        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        cmd : str
            Command to be submitted in shell script. Example:
                'python -m sup3r.cli <module_name> -c <config_file>'
        """
        cls.check_module_name(module_name)

        name = ctx.obj['NAME']
        out_dir = ctx.obj['OUT_DIR']
        subprocess_manager = SubprocessManager
        status = Status.retrieve_job_status(out_dir,
                                            module=module_name,
                                            job_name=name)
        msg = f'sup3r {module_name} CLI failed to submit jobs!'
        if status == 'successful':
            msg = (f'Job "{name}" is successful in status json found in '
                   f'"{out_dir}", not re-running.')
        elif 'fail' not in str(status).lower() and status is not None:
            msg = (f'Job "{name}" was found with status "{status}", not '
                   'resubmitting')
        else:
            logger.info(f'Running sup3r {module_name} locally with job '
                        f'name "{name}".')
            Status.add_job(out_dir, module=module_name, job_name=name,
                           replace=True)
            subprocess_manager.submit(cmd)
            msg = (f'Completed sup3r {module_name} job "{name}".')

        click.echo(msg)
        logger.info(msg)

    @classmethod
    def add_status_cmd(cls, config, module_name, cmd):
        """Append status file command to command for executing given module

        Parameters
        ----------
        config : dict
            sup3r config with all necessary args and kwargs to run given
            module.
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        cmd : str
            String including command to execute given module.

        Returns
        -------
        cmd : str
            Command string with status file command included if job_name is
            not None
        """
        job_name = config.get('job_name', None)
        status_dir = config.get('status_dir', None)
        if job_name is not None and status_dir is not None:
            status_file_arg_str = f'"{status_dir}", '
            status_file_arg_str += f'module="{module_name}", '
            status_file_arg_str += f'job_name="{job_name}", '
            status_file_arg_str += 'attrs=job_attrs'

            cmd += ('job_attrs = {};\n'.format(json.dumps(config)
                                               .replace("null", "None")
                                               .replace("false", "False")
                                               .replace("true", "True")))
            cmd += 'job_attrs.update({"job_status": "successful"});\n'
            cmd += 'job_attrs.update({"time": t_elap});\n'
            cmd += f"Status.make_job_file({status_file_arg_str})"

        return cmd
