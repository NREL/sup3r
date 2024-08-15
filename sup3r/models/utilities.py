"""Utilities shared across the `sup3r.models` module"""

import logging
import sys
import threading

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class TrainingSession:
    """Wrapper to gracefully exit batch handler thread during training, upon a
    keyboard interruption."""

    def __init__(self, batch_handler, model, **kwargs):
        """
        Parameters
        ----------
        batch_handler: BatchHandler
            Batch iterator
        model: Sup3rGan
            Gan model to run in new thread
        **kwargs : dict
            Model keyword args
        """
        self.batch_handler = batch_handler
        self.model = model
        self.kwargs = kwargs

    def run(self):
        """Wrap model.train()."""
        model_thread = threading.Thread(
            target=self.model.train,
            args=(self.batch_handler,),
            kwargs=self.kwargs,
        )
        try:
            logger.info('Starting training session.')
            self.batch_handler.start()
            model_thread.start()
        except KeyboardInterrupt:
            logger.info('Ending training session.')
            self.batch_handler.stop()
            model_thread.join()
            sys.exit()
        except Exception as e:
            logger.info('Ending training session. %s', e)
            self.batch_handler.stop()
            model_thread.join()
            sys.exit()

        logger.info('Finished training')
        self.batch_handler.stop()
        model_thread.join()


def st_interp(low, s_enhance, t_enhance, t_centered=False):
    """Spatiotemporal bilinear interpolation for low resolution field on a
    regular grid. Used to provide baseline for comparison with gan output

    Parameters
    ----------
    low : ndarray
        Low resolution field to interpolate.
        (spatial_1, spatial_2, temporal)
    s_enhance : int
        Factor by which to enhance the spatial domain
    t_enhance : int
        Factor by which to enhance the temporal domain
    t_centered : bool
        Flag to switch time axis from time-beginning (Default, e.g.
        interpolate 00:00 01:00 to 00:00 00:30 01:00 01:30) to
        time-centered (e.g. interp 01:00 02:00 to 00:45 01:15 01:45 02:15)

    Returns
    -------
    ndarray
        Spatiotemporally interpolated low resolution output
    """
    assert len(low.shape) == 3, 'Input to st_interp must be 3D array'
    msg = 'Input to st_interp cannot include axes with length 1'
    assert not any(s <= 1 for s in low.shape), msg

    lr_y, lr_x, lr_t = low.shape
    hr_y, hr_x, hr_t = lr_y * s_enhance, lr_x * s_enhance, lr_t * t_enhance

    # assume outer bounds of mesh (0, 10) w/ points on inside of that range
    y = np.arange(0, 10, 10 / lr_y) + 5 / lr_y
    x = np.arange(0, 10, 10 / lr_x) + 5 / lr_x

    # remesh (0, 10) with high res spacing
    new_y = np.arange(0, 10, 10 / hr_y) + 5 / hr_y
    new_x = np.arange(0, 10, 10 / hr_x) + 5 / hr_x

    t = np.arange(0, 10, 10 / lr_t)
    new_t = np.arange(0, 10, 10 / hr_t)
    if t_centered:
        t += 5 / lr_t
        new_t += 5 / hr_t

    # set RegularGridInterpolator to do extrapolation
    interp = RegularGridInterpolator(
        (y, x, t), low, bounds_error=False, fill_value=None
    )

    # perform interp
    X, Y, T = np.meshgrid(new_x, new_y, new_t)
    return interp((Y, X, T))
