"""Classes for handling training with h5 data as high res (usually WTK) and
ERA5 as low res"""
import logging

import numpy as np
import xesmf as xe

from sup3r.preprocessing.batch_handling import Batch, BatchHandler
from sup3r.preprocessing.data_handling import DataHandlerH5, DataHandlerNC

logger = logging.getLogger(__name__)


# pylint: disable=unsubscriptable-object
class DataHandlerH5withERA(DataHandlerNC):
    """Batch handling class for h5 data as high res (usually WTK) and ERA5 as
    low res"""

    def __init__(
        self, *args, era_kwargs, s_enhance=15, t_enhance=12, **kwargs
    ):
        """Initialize data handler using data handlers for h5 data and era5
        data

        Parameters
        ----------
        *args : tuple
            Same as arguments for base data handler
        era_kwargs : dict
            Dictionary of kwargs for era data handler (DataHandlerNC)
        s_enhance : int
            Spatial enhancement factor
        t_enhance : int
            Temporal enhancement factor
        **kwargs : dict
            Same as kwargs for base data handler
        """
        self.h5_dh = DataHandlerH5(*args, **kwargs)

        self.era_dh = DataHandlerNC(**era_kwargs, features=self.features)

        era_grid = {'lat': self.era_dh.lat_lon[..., 0],
                    'lon': self.era_dh.lat_lon[..., 1]}
        h5_grid = {'lat': self.h5_dh.lat_lon[..., 0],
                   'lon': self.h5_dh.lat_lon[..., 1]}

        self.regridder = xe.Regridder(era_grid, h5_grid, method='bilinear')

        self.data = self.regrid_era()
        self.hr_data = self.h5_dh.data
        self.lat_lon = self.h5_dh.lat_lon

        self.s_enhance = s_enhance
        self.t_enhance = t_enhance

    def regrid_feature(self, fidx):
        """Regrid ERA5 feature data to h5 data grid

        Parameters
        ----------
        fidx : int
            Feature index

        Returns
        -------
        out : ndarray
            Array of regridded ERA5 data
            (spatial_1, spatial_2, temporal)
        """
        out = np.concatenate(
            [
                self.regridder(self.era_dh.data[..., i, fidx])[..., np.newaxis]
                for i in range(len(self.era_dh.time_index))
            ],
            axis=-1,
        )
        return out

    def regrid_era(self):
        """Regrid ERA5 data for all requested features

        Returns
        -------
        out : ndarray
            Array of regridded ERA5 data with all features
            (spatial_1, spatial_2, temporal, n_features)
        """
        return np.concatenate(
            [
                self.regrid_feature(i)[..., np.newaxis]
                for i in range(len(self.features))
            ],
            axis=-1,
        )

    def get_next(self):
        """Get next h5 + era pair. Gets random spatiotemporal sample for h5
        data and then uses enhancement factors to subsample
        interpolated/regridded ERA5 data for same spatiotemporal extent.

        Returns
        -------
        hr_data : ndarray
            Array of high resolution data with each feature equal in shape to
            sample_shape
        lr_data : ndarray
            Array of low resolution data with each feature equal in shape to
            sample_shape // (s_enhance or t_enhance)
        """
        hr_obs_idx = self.get_observation_index()
        lr_obs_idx = []
        for s in hr_obs_idx[:2]:
            lr_obs_idx.append(slice(s.start, s.end, self.s_enhance))
        for s in hr_obs_idx[2:-1]:
            lr_obs_idx.append(slice(s.start, s.end, self.t_enhance))
        lr_obs_idx.append(hr_obs_idx[-1])

        return self.hr_data[hr_obs_idx], self.data[lr_obs_idx]


class BatchHandlerH5withERA(BatchHandler):
    """Batch handling class for h5 data as high res (usually WTK) and ERA5 as
    low res"""

    BATCH_CLASS = Batch

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        """Get the next iterator output.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res and batch.high_res attributes
            with the appropriate subsampling of interpolated ERA.
        """
        self.current_batch_indices = []
        if self._i < self.n_batches:
            handler_index = np.random.randint(0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros(
                (
                    self.batch_size,
                    self.sample_shape[0],
                    self.sample_shape[1],
                    self.sample_shape[2],
                    self.shape[-1],
                ),
                dtype=np.float32,
            )
            low_res = np.zeros(
                (
                    self.batch_size,
                    self.sample_shape[0] // self.s_enhance,
                    self.sample_shape[1] // self.s_enhance,
                    self.sample_shape[2] // self.t_enhance,
                    self.shape[-1],
                ),
                dtype=np.float32,
            )

            for i in range(self.batch_size):
                high_res[i, ...], low_res[i, ...] = handler.get_next()
                self.current_batch_indices.append(handler.current_obs_index)

            batch = self.BATCH_CLASS(low_res=low_res, high_res=high_res)

            self._i += 1
            return batch
        else:
            raise StopIteration
