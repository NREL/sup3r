# -*- coding: utf-8 -*-
"""Module to plot feature output from forward passes for visual inspection"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import glob
import json
from datetime import datetime as dt

import rex
from rex.utilities.fun_utils import get_fun_call_str
from concurrent.futures import ThreadPoolExecutor, as_completed

from sup3r.utilities import ModuleName


logger = logging.getLogger(__name__)


class Sup3rVisualQa:
    """Module to plot features for visual qa"""

    def __init__(self, file_paths, out_pattern, features, time_step=10,
                 spatial_slice=slice(None), source_handler_class=None,
                 workers=None, **kwargs):
        """
        Parameters
        ----------
        file_paths : list | str
            Specifies the files to use for the plotting routine. This is either
            a list of h5 files generated by the forward pass module or a string
            pointing to h5 forward pass output which can be parsed by glob.glob
        out_pattern : str
            The pattern to use for naming the plot figures. This must include
            {feature} and {index} so output files can be named with
            out_pattern.format(feature=feature, index=index).
            e.g. outfile_{feature}_{index}.png. The number of plot figures is
            determined by the time_index of the h5 files and the time_step
            argument. The index key refers to the plot file index from the list
            of all plot files generated.
        features : list
            List of features to plot from the h5 files provided.
        time_step : int
            Number of timesteps to average over for a single plot figure.
        spatial_slice : slice
            Slice specifying the spatial range to plot. This can include a
            step > 1 to speed up plotting.
        source_handler_class : str | None
            Name of the class to use for h5 input files. If None this defaults
            to MultiFileResource.
        workers : int | None
            Max number of workers to use for plotting. If workers=1 then all
            plots will be created in serial.
        **kwargs : dict
            Dictionary of kwargs passed to matplotlib.pyplot.scatter().
        """

        self.features = features
        self.out_pattern = out_pattern
        self.time_step = time_step
        self.spatial_slice = (spatial_slice if isinstance(spatial_slice, slice)
                              else slice(*spatial_slice))
        self.file_paths = (file_paths if isinstance(file_paths, list)
                           else glob.glob(file_paths))
        self.workers = workers
        self.kwargs = kwargs
        self.res_handler = source_handler_class or 'MultiFileResource'
        self.res_handler = getattr(rex, self.res_handler)

    def run(self):
        """
        Create plot figures for all the features in self.features. For each
        feature there will be n_files created, where n_files is the number of
        timesteps in the h5 files provided divided by self.time_step.
        """
        with self.res_handler(self.file_paths) as res:
            time_index = res.time_index
            n_files = len(time_index[::self.time_step])
            time_slices = np.array_split(np.arange(len(time_index)), n_files)
            time_slices = [slice(s[0], s[-1] + 1) for s in time_slices]

            if self.workers == 1:
                self._serial_figure_plots(res, time_index, time_slices,
                                          self.spatial_slice)
            else:
                self._parallel_figure_plots(res, time_index, time_slices,
                                            self.spatial_slice)

    def _serial_figure_plots(self, res, time_index, time_slices,
                             spatial_slice):
        """Plot figures in parallel with max_workers=self.workers

        Parameters
        ----------
        res : MultiFileResourceX
            Resource handler for the provided h5 files
        time_index : pd.DateTimeIndex
            The time index for the provided h5 files
        time_slices : list
             List of slices specifying all the time ranges to average and plot
        spatial_slice : slice
             Slice specifying the spatial range to plot
        """
        for feature in self.features:
            for i, t_slice in enumerate(time_slices):
                out_file = self.out_pattern.format(feature=feature,
                                                   index=i)
                self.plot_figure(res, time_index, feature, t_slice,
                                 spatial_slice, out_file)

    def _parallel_figure_plots(self, res, time_index, time_slices,
                               spatial_slice):
        """Plot figures in parallel with max_workers=self.workers

        Parameters
        ----------
        res : MultiFileResourceX
            Resource handler for the provided h5 files
        time_index : pd.DateTimeIndex
            The time index for the provided h5 files
        time_slices : list
             List of slices specifying all the time ranges to average and plot
        spatial_slice : slice
             Slice specifying the spatial range to plot
        """
        futures = {}
        now = dt.now()
        n_files = len(time_slices) * len(self.features)
        with ThreadPoolExecutor(max_workers=self.workers) as exe:
            for feature in self.features:
                for i, t_slice in enumerate(time_slices):
                    out_file = self.out_pattern.format(feature=feature,
                                                       index=i)
                    future = exe.submit(self.plot_figure, res, time_index,
                                        feature, t_slice, spatial_slice,
                                        out_file)
                    futures[future] = out_file

            logger.info(f'Started plotting {n_files} files '
                        f'in {dt.now() - now}.')

            for i, future in enumerate(as_completed(futures)):
                try:
                    future.result()
                except Exception as e:
                    msg = (f'Error making plot {futures[future]}.')
                    logger.exception(msg)
                    raise RuntimeError(msg) from e
                logger.debug(f'{i+1} out of {n_files} plots created.')

    def plot_figure(self, res, time_index, feature, t_slice, s_slice,
                    out_file):
        """Plot temporal average for the given feature and with the time range
        specified by t_slice

        Parameters
        ----------
        res : MultiFileResourceX
            Resource handler for the provided h5 files
        time_index : pd.DateTimeIndex
            The time index for the provided h5 files
        feature : str
            The feature to plot
        t_slice : slice
            The slice specifying the time range to average and plot
        s_slice : slice
            The slice specifying the spatial range to plot.
        out_file : str
            Name of the output plot file
        """
        start_time = time_index[t_slice.start]
        stop_time = time_index[t_slice.stop - 1]
        logger.info(f'Plotting time average for {feature} from '
                    f'{start_time} to {stop_time}.')
        fig = plt.figure()
        title = f'{feature}: {start_time} - {stop_time}'
        plt.suptitle(title)
        plt.scatter(res.meta.longitude, res.meta.latitude,
                    c=np.mean(res[feature, t_slice, s_slice], axis=0),
                    **self.kwargs)
        plt.colorbar()
        fig.savefig(out_file)
        plt.close()
        logger.info(f'Saved figure {out_file}')

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to initialize Sup3rVisualQa and execute the
        Sup3rVisualQa.run() method based on an input config

        Parameters
        ----------
        config : dict
            sup3r QA config with all necessary args and kwargs to
            initialize Sup3rVisualQa and execute Sup3rVisualQa.run()
        """
        import_str = 'import time;\n'
        import_str += 'from reV.pipeline.status import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += 'from sup3r.qa.visual_qa import Sup3rVisualQa;\n'

        qa_init_str = get_fun_call_str(cls, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')

        log_arg_str = (f'"sup3r", log_level="{log_level}"')
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (f"python -c \'{import_str}\n"
               "t0 = time.time();\n"
               f"logger = init_logger({log_arg_str});\n"
               f"qa = {qa_init_str};\n"
               "qa.run();\n"
               "t_elap = time.time() - t0;\n")

        job_name = config.get('job_name', None)
        if job_name is not None:
            status_dir = config.get('status_dir', None)
            status_file_arg_str = f'"{status_dir}", '
            status_file_arg_str += f'module="{ModuleName.VISUAL_QA}", '
            status_file_arg_str += f'job_name="{job_name}", '
            status_file_arg_str += 'attrs=job_attrs'

            cmd += ('job_attrs = {};\n'.format(json.dumps(config)
                                               .replace("null", "None")
                                               .replace("false", "False")
                                               .replace("true", "True")))
            cmd += 'job_attrs.update({"job_status": "successful"});\n'
            cmd += 'job_attrs.update({"time": t_elap});\n'
            cmd += f'Status.make_job_file({status_file_arg_str})'

        cmd += (";\'\n")

        return cmd.replace('\\', '/')