"""Base data handling classes.
@author: bbenton
"""
import logging

import numpy as np

from sup3r.preprocessing.data_loading.abstract import AbstractLoader

np.random.seed(42)

logger = logging.getLogger(__name__)


class LazyLoader(AbstractLoader):
    """Base lazy loader. Loads precomputed netcdf files (usually from
    a DataHandler.to_netcdf() call after populating DataHandler.data) to create
    batches on the fly during training without previously loading to memory."""

    def get_observation(self, obs_index):
        """Get observation/sample. Should return a single sample from the
        underlying data with shape (spatial_1, spatial_2, temporal,
        features)."""

        out = self.data.isel(
            south_north=obs_index[0],
            west_east=obs_index[1],
            time=obs_index[2],
        )

        if self._mode == 'lazy':
            out = out.compute()

        out = out.to_dataarray().values
        return np.transpose(out, axes=(2, 3, 1, 0))


