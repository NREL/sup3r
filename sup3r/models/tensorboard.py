"""Abstract class defining the required interface for Sup3r model subclasses"""

import logging
import os

import tensorflow as tf

from sup3r.utilities.utilities import Timer

logger = logging.getLogger(__name__)


class TensorboardMixIn:
    """MixIn class for tensorboard logging and profiling."""

    def __init__(self):
        self._tb_writer = None
        self._tb_log_dir = None
        self._write_tb_profile = False
        self._total_batches = None
        self._history = None
        self.timer = Timer()

    @property
    def total_batches(self):
        """Record of total number of batches for logging."""
        if self._total_batches is None and self._history is None:
            self._total_batches = 0
        elif self._history is None and 'total_batches' in self._history:
            self._total_batches = self._history['total_batches'].values[-1]
        elif self._total_batches is None and self._history is not None:
            self._total_batches = 0
        return self._total_batches

    @total_batches.setter
    def total_batches(self, value):
        """Set total number of batches."""
        self._total_batches = value

    def dict_to_tensorboard(self, entry):
        """Write data to tensorboard log file. This is usually a loss_details
        dictionary.

        Parameters
        ----------
        entry: dict
            Dictionary of values to write to tensorboard log file
        """
        if self._tb_writer is not None:
            with self._tb_writer.as_default():
                for name, value in entry.items():
                    if isinstance(value, str):
                        tf.summary.text(name, value, self.total_batches)
                    else:
                        tf.summary.scalar(name, value, self.total_batches)

    def profile_to_tensorboard(self, name):
        """Write profile data to tensorboard log file.

        Parameters
        ----------
        name : str
            Tag name to use for profile info
        """
        if self._tb_writer is not None and self._write_tb_profile:
            with self._tb_writer.as_default():
                tf.summary.trace_export(
                    name=name,
                    step=self.total_batches,
                    profiler_outdir=self._tb_log_dir,
                )

    def _init_tensorboard_writer(self, out_dir):
        """Initialize the ``tf.summary.SummaryWriter`` to use for writing
        tensorboard compatible log files.

        Parameters
        ----------
        out_dir : str
            Standard out_dir where model epochs are saved. e.g. './gan_{epoch}'
        """
        tb_log_pardir = os.path.abspath(os.path.join(out_dir, os.pardir))
        self._tb_log_dir = os.path.join(tb_log_pardir, 'logs')
        os.makedirs(self._tb_log_dir, exist_ok=True)
        self._tb_writer = tf.summary.create_file_writer(self._tb_log_dir)
