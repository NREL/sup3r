"""Quantile Delta Mapping
"""

import logging
from glob import glob

import json
import h5py
import numpy as np

from .distribution import EmpiricalDistribution
from .bias_calc import DataRetrievalBase

logger = logging.getLogger(__name__)


class QuantileDeltaMapping(DataRetrievalBase):
    def __init__(self, base_fps, bias_fps, bias_fut_fps, *args, **kwargs):
        self.NQ = 51
        super().__init__(base_fps, bias_fps, *args, **kwargs)

        self.bias_fut_fps = bias_fps

        if isinstance(self.bias_fut_fps, str):
            self.bias_fut_fps = sorted(glob(self.bias_fut_fps))

        self.bias_fut_dh = self.bias_handler(self.bias_fut_fps,
                                             [self.bias_feature],
                                             target=self.target,
                                             shape=self.shape,
                                             val_split=0.0,
                                             **self.bias_handler_kwargs)

        assert np.allclose(self.bias_dh.lat_lon, self.bias_fut_dh.lat_lon)


    def _init_out(self):
        """Initialize output arrays"""
        keys = [f'bias_{self.bias_feature}_CDF',
                f'bias_fut_{self.bias_feature}_CDF',
                f'base_{self.base_dset}_CDF',
                ]
        self.out = {k: np.full((*self.bias_gid_raster.shape, self.NQ),
                               np.nan, np.float32)
                    for k in keys}

    def get_base_data(self, base_gid, daily_reduction):
        """Get base data for given GID

        For now, there is no need to change most of the arguments, thus
        this wrapper simplifies the DataRetrievalBase.get_base_data()
        by using its context.

        Parameters
        ----------

        Return
        ------

        """
        base_data, _ = super().get_base_data(self.base_fps,
                                             self.base_dset,
                                             base_gid,
                                             self.base_handler,
                                             base_handler_kwargs=None,
                                             daily_reduction=daily_reduction,
                                             decimals=self.decimals,
                                             base_dh_inst=self.base_dh
                                             )
        return base_data

    def run(self, daily_reduction='avg'):
        """

        Reminders:
        - Options between relative or absolute correction
        """
        logger.debug("Starting linear correction calculation...")

        logger.debug('Running serial calculation.')
        self.NQ = 51
        keys = ["base_CDF", "bias_CDF", "bias_fut_CDF"]
        self.out = {k: np.full((*self.bias_gid_raster.shape, self.NQ),
                               np.nan, np.float32)
                    for k in keys}

        # Sampling fixed to linear for now
        self.meta["sampling"] = "linear"
        for i, bias_gid in enumerate(self.bias_meta.index):
            raster_loc = np.where(self.bias_gid_raster == bias_gid)
            _, base_gid = self.get_base_gid(bias_gid)

            if not base_gid.any():
                self.bad_bias_gids.append(bias_gid)
                logger.debug(f"No base data for bias_gid: {bias_gid}. "
                             "Adding it to bad_bias_gids")
            else:
                bias_data = self.get_bias_data(bias_gid)
                bias_fut_data = self.get_bias_data(bias_gid, self.bias_fut_dh)
                base_data = self.get_base_data(base_gid, daily_reduction)

                D_base = EmpiricalDistribution.from_fit(base_data, self.NQ)
                self.out[f'base_{self.base_dset}_CDF'][raster_loc] = D_base.cut_point

                D_bias = EmpiricalDistribution.from_fit(bias_data, self.NQ)
                self.out[f'bias_{self.bias_feature}_CDF'][raster_loc] = D_bias.cut_point
                D_bias_fut = EmpiricalDistribution.from_fit(bias_fut_data, self.NQ)
                self.out[f'bias_fut_{self.bias_feature}_CDF'][raster_loc] = D_bias_fut.cut_point

            logger.info('Completed bias calculations for {} out of {} '
                            'sites'.format(i + 1, len(self.bias_meta)))

            # write_outputs
            filename = 'dist.h5'
            with h5py.File(fp_out, 'w') as f:
                lat = self.bias_dh.lat_lon[..., 0]
                lon = self.bias_dh.lat_lon[..., 1]
                f.create_dataset('latitude', data=lat)
                f.create_dataset('longitude', data=lon)
                for dset, data in self.out.items():
                    f.create_dataset(dset, data=data)

                for k, v in self.meta.items():
                    f.attrs[k] = json.dumps(v)
                logger.info(
                    'Wrote quantiles to file: {}'.format(filename))