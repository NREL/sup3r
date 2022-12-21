
# -*- coding: utf-8 -*-
"""
Sup3r base CLI class.
"""
import click
import logging
import os

from reV.pipeline.status import Status

from rex.utilities.execution import SubprocessManager
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_mult

from sup3r.version import __version__
from sup3r.pipeline.config import BaseConfig


logger = logging.getLogger(__name__)


class BaseCLI:
    """Base CLI class which can be used to construct CLI for sup3r modules"""

    def __init__(self, module_name, module_class):
        """
        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        module_class : Object
            Class object to use for getting node command. For example, this can
            be sup3r.qa.qa.Sup3rQa, which will then be used to call
            Sup3rQa.get_node_cmd(config).
        """
        self.module_name = module_name
        self.module_class = module_class
        self.underscore_name = module_name.replace('-', '_')

    def from_config(self, ctx, config_file, verbose):
        """Run sup3r module from a config file."""
        ctx.ensure_object(dict)
        ctx.obj['VERBOSE'] = verbose
        status_dir = os.path.dirname(os.path.abspath(config_file))
        ctx.obj['OUT_DIR'] = status_dir
        config = BaseConfig(config_file)
        config['status_dir'] = status_dir
        config_verbose = config.get('log_level', 'INFO')
        config_verbose = (config_verbose == 'DEBUG')
        verbose = any([verbose, config_verbose, ctx.obj['VERBOSE']])

        init_mult(f'sup3r_{self.underscore_name}',
                  os.path.join(status_dir, 'logs/'),
                  modules=[__name__, 'sup3r'], verbose=verbose)

        exec_kwargs = config.get('execution_control', {})
        log_pattern = config.get('log_pattern', None)
        if log_pattern is not None:
            os.makedirs(os.path.dirname(log_pattern), exist_ok=True)
            if '.log' not in log_pattern:
                log_pattern += '.log'
            if '{node_index}' not in log_pattern:
                log_pattern = log_pattern.replace('.log', '_{node_index}.log')

        hardware_option = exec_kwargs.pop('option', 'local')
        exec_kwargs['stdout_path'] = os.path.join(status_dir, 'stdout/')
        logger.debug('Found execution kwargs: {}'.format(exec_kwargs))
        logger.debug('Hardware run option: "{}"'.format(hardware_option))

        name = f'sup3r_{self.underscore_name}'
        name += '_{}'.format(os.path.basename(status_dir))
        job_name = config.get('job_name', None)
        if job_name is not None:
            name = job_name
        ctx.obj['NAME'] = name
        config['job_name'] = name
        config['status_dir'] = status_dir

        cmd = self.module_class.get_node_cmd(config)

        cmd_log = '\n\t'.join(cmd.split('\n'))
        logger.debug(f'Running command:\n\t{cmd_log}')

        if hardware_option.lower() in ('eagle', 'slurm'):
            self.kickoff_slurm_job(ctx, cmd, **exec_kwargs)
        else:
            self.kickoff_local_job(ctx, cmd)

    def kickoff_slurm_job(self, ctx, cmd, alloc='sup3r', memory=None,
                          walltime=4, feature=None, stdout_path='./stdout/'):
        """Run sup3r module on HPC via SLURM job submission.

        Parameters
        ----------
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

        name = ctx.obj['NAME']
        out_dir = ctx.obj['OUT_DIR']
        slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
        if slurm_manager is None:
            slurm_manager = SLURM()
            ctx.obj['SLURM_MANAGER'] = slurm_manager

        status = Status.retrieve_job_status(out_dir,
                                            module=self.module_name,
                                            job_name=name,
                                            hardware='slurm',
                                            subprocess_manager=slurm_manager)

        msg = f'sup3r {self.module_name} CLI failed to submit jobs!'
        if status == 'successful':
            msg = (f'Job "{name}" is successful in status json found in '
                   f'"{out_dir}", not re-running.')
        elif 'fail' not in str(status).lower() and status is not None:
            msg = (f'Job "{name}" was found with status "{status}", not '
                   'resubmitting')
        else:
            logger.info(f'Running sup3r {self.module_name} on SLURM with node '
                        f'name "{name}".')
            out = slurm_manager.sbatch(cmd,
                                       alloc=alloc,
                                       memory=memory,
                                       walltime=walltime,
                                       feature=feature,
                                       name=name,
                                       stdout_path=stdout_path)[0]
            if out:
                msg = (f'Kicked off sup3r {self.module_name} job "{name}" '
                       f'(SLURM jobid #{out}).')

            # add job to sup3r status file.
            Status.add_job(out_dir, module=self.module_name,
                           job_name=name, replace=True,
                           job_attrs={'job_id': out, 'hardware': 'slurm'})

        click.echo(msg)
        logger.info(msg)

    def kickoff_local_job(self, ctx, cmd):
        """Run sup3r module locally.

        Parameters
        ----------
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        cmd : str
            Command to be submitted in shell script. Example:
                'python -m sup3r.cli <module_name> -c <config_file>'
        """

        name = ctx.obj['NAME']
        out_dir = ctx.obj['OUT_DIR']
        subprocess_manager = SubprocessManager
        status = Status.retrieve_job_status(out_dir,
                                            module=self.module_name,
                                            job_name=name)
        msg = f'sup3r {self.module_name} CLI failed to submit jobs!'
        if status == 'successful':
            msg = (f'Job "{name}" is successful in status json found in '
                   f'"{out_dir}", not re-running.')
        elif 'fail' not in str(status).lower() and status is not None:
            msg = (f'Job "{name}" was found with status "{status}", not '
                   'resubmitting')
        else:
            logger.info(f'Running sup3r {self.module_name} locally with job '
                        f'name "{name}".')
            Status.add_job(out_dir, module=self.module_name, job_name=name,
                           replace=True)
            subprocess_manager.submit(cmd)
            msg = (f'Completed sup3r {self.module_name} job "{name}".')

        click.echo(msg)
        logger.info(msg)
