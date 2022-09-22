# -*- coding: utf-8 -*-
"""Utilities to calculate the bias correction factors for biased data that is
going to be fed into the sup3r downscaling models. This is typically used to
bias correct GCM data vs. some historical record like the WTK or NSRDB."""
import h5py
import logging
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import ks_2samp
from concurrent.futures import ProcessPoolExecutor, as_completed
import rex
from sup3r.utilities.utilities import nn_fill_array
import sup3r.preprocessing.data_handling


logger = logging.getLogger(__name__)


class DataRetrievalBase:
    """Base class to handle data retrieval for the biased data and the
    baseline data
    """

    def __init__(self, base_fps, bias_fps, base_dset, bias_features,
                 target, shape,
                 base_handler='Resource', bias_handler='DataHandlerNCforCC',
                 bias_handler_kwargs=None):
        """
        Parameters
        ----------
        base_fps : list | str
            One or more baseline .h5 filepaths representing non-biased data to
            use to correct the biased dataset. This is typically several years
            of WTK or NSRDB files.
        bias_fps : list | str
            One or more biased .nc or .h5 filepaths representing the biased
            data to be corrected based on the baseline data. This is typically
            several years of GCM .nc files.
        base_dset : str
            A single dataset from the base_fps to retrieve.
        bias_features : str | list
            This is the biased features from bias_fps to retrieve. This should
            be a single feature name, or in the case of windspeed this will be
            the two north/east components (e.g. bias_features=[U_100m, V_100m]
            corresponding to base_dset=windspeed_100m)
        target : tuple
            (lat, lon) lower left corner of raster to retrieve from bias_fps.
        shape : tuple
            (rows, cols) grid size to retrieve from bias_fps.
        base_handler : str
            Name of rex resource handler class to be retrieved from the rex
            library.
        bias_handler : str
            Name of the bias data handler class to be retrieved from the
            sup3r.preprocessing.data_handling library.
        """
        self.base_fps = base_fps
        self.bias_fps = bias_fps
        self.base_dset = base_dset
        self.bias_features = bias_features
        self.target = target
        self.shape = shape
        bias_handler_kwargs = bias_handler_kwargs or {}

        if isinstance(self.base_fps, str):
            self.base_fps = [self.base_fps]
        if isinstance(self.bias_fps, str):
            self.bias_fps = [self.bias_fps]
        if isinstance(self.bias_features, str):
            self.bias_features = [self.bias_features]

        if 'windspeed' in self.base_dset and len(self.bias_features) != 2:
            msg = ('Base windspeed dataset of "{}" needs exactly two features '
                   'for the bias data corresponding to the U and V '
                   'components, but received: {}'
                   .format(self.base_dset, self.bias_features))
            logger.error(msg)
            raise RuntimeError(msg)

        elif 'windspeed' not in self.base_dset and len(self.bias_features) != 1:
            msg = ('If base dataset is not windspeed, cannot handle more than '
                   'one feature from the bias data, but received: {}'
                   .format(self.bias_features))
            logger.error(msg)
            raise RuntimeError(msg)

        self.base_handler = getattr(rex, base_handler)
        self.bias_handler = getattr(sup3r.preprocessing.data_handling,
                                    bias_handler)

        with self.base_handler(self.base_fps[0]) as res:
            self.base_meta = res.meta
            self.base_tree = KDTree(self.base_meta[['latitude', 'longitude']])

        self.bias_dh = self.bias_handler(self.bias_fps, self.bias_features,
                                         target=self.target, shape=self.shape,
                                         val_split=0.0, **bias_handler_kwargs)

        lats = self.bias_dh.lat_lon[..., 0].flatten()
        lons = self.bias_dh.lat_lon[..., 1].flatten()
        self.bias_meta = pd.DataFrame({'latitude': lats, 'longitude': lons})

        raster_shape = self.bias_dh.lat_lon[..., 0].shape
        self.bias_tree = KDTree(self.bias_meta[['latitude', 'longitude']])
        self.bias_gid_raster = np.arange(lats.size)
        self.bias_gid_raster = self.bias_gid_raster.reshape(raster_shape)
        logger.info('Finished initializing DataRetrievalBase.')

    @staticmethod
    def compare_dists(base_data, bias_data, adder=0, scalar=1):
        """Compare two distributions using the two-sample Kolmogorov-Smirnov.
        When the output is minimized, the two distributions are similar.

        Parameters
        ----------
        base_data : np.ndarray
            1D array of base data observations.
        bias_data : np.ndarray
            1D array of biased data observations.
        adder : float
            Factor to adjust the biased data before comparing distributions:
            bias_data * scalar + adder
        scalar : float
            Factor to adjust the biased data before comparing distributions:
            bias_data * scalar + adder

        Returns
        -------
        out : float
            KS test statistic
        """
        out = ks_2samp(base_data, bias_data * scalar + adder)
        return out.statistic

    def get_base_gid(self, bias_gid, knn):
        """Get one or more base gid(s) corresponding to a bias gid.

        Parameters
        ----------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.
        knn : int
            Number of nearest neighbors to aggregate from the base data when
            comparing to a single site from the bias data.

        Returns
        -------
        dist : np.ndarray
            Array of nearest neighbor distances with length == knn
        base_gid : np.ndarray
            Array of base gids that are the nearest neighbors of bias_gid with
            length == knn
        """
        coord = self.bias_meta.loc[bias_gid, ['latitude', 'longitude']]
        dist, base_gid = self.base_tree.query(coord, k=knn)
        return dist, base_gid

    def get_data_pair(self, bias_gid, knn, daily_avg=True):
        """Get base and bias data observations based on a single bias gid.

        Parameters
        ----------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.
        knn : int
            Number of nearest neighbors to aggregate from the base data when
            comparing to a single site from the bias data.
        daily_avg : bool
            Flag to do temporal daily averaging of the base data.

        Returns
        -------
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
            and possibly daily-averaged as well.
        bias_data : np.ndarray
            1D array of temporal data at the requested gid.
        dist : np.ndarray
            Array of nearest neighbor distances with length == knn
        """
        dist, base_gid = self.get_base_gid(bias_gid, knn)
        bias_data = self.get_bias_data(bias_gid)
        base_data = self.get_base_data(self.base_fps, self.base_dset, base_gid,
                                       self.base_handler, daily_avg=daily_avg)
        return base_data, bias_data, dist

    def get_bias_data(self, bias_gid):
        """Get data from the biased data source for a single gid

        Parameters
        ----------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.

        Returns
        -------
        bias_data : np.ndarray
            1D array of temporal data at the requested gid.
        """
        idx = np.where(self.bias_gid_raster == bias_gid)
        bias_data = self.bias_dh.data[idx][0]
        if bias_data.shape[-1] > 1 and 'windspeed' in self.base_dset:
            bias_data = np.hypot(bias_data[:, 0], bias_data[:, 1])
        elif bias_data.shape[-1] == 1:
            bias_data = bias_data[:, 0]
        else:
            msg = ('Found a weird number of feature channels for the bias '
                   'data retrieval: {}'.format(bias_data.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        return bias_data

    @staticmethod
    def get_base_data(base_fps, base_dset, base_gid, base_handler,
                      daily_avg=True):
        """Get data from the baseline data source, possibly for many high-res
        base gids corresponding to a single coarse low-res bias gid.

        Parameters
        ----------
        base_fps : list | str
            One or more baseline .h5 filepaths representing non-biased data to
            use to correct the biased dataset. This is typically several years
            of WTK or NSRDB files.
        base_dset : str
            A single dataset from the base_fps to retrieve.
        base_gid : int | np.ndarray
            One or more spatial gids to retrieve from base_fps. The data will
            be spatially averaged across all of these sites.
        base_handler : rex.Resource
            A rex data handler similar to rex.Resource
        daily_avg : bool
            Flag to do temporal daily averaging of the base data.

        Returns
        -------
        out : np.ndarray
            1D array of base data spatially averaged across the base_gid input
            and possibly daily-averaged as well.
        """
        out = []
        for fp in base_fps:
            with base_handler(fp) as res:
                base_ti = res.time_index
                base_data = res[base_dset, :, base_gid]
                if len(base_data.shape) == 2:
                    base_data = base_data.mean(axis=1)

                if daily_avg:
                    slices = [np.where(base_ti.date == date)
                              for date in sorted(set(base_ti.date))]
                    base_data = np.array([base_data[s0].mean()
                                          for s0 in slices])

            out.append(base_data)

        return np.hstack(out)


class LinearCorrection(DataRetrievalBase):
    """Calculate linear correction *scalar +adder factors to bias correct data
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar = None
        self.adder = None
        logger.info('Finished initializing LinearCorrection.')

    @classmethod
    def _run_single(cls, bias_data, base_fps, base_dset, base_gid,
                    base_handler, daily_avg):
        """Find the nominal scalar + adder combination to bias correct data
        at a single site"""

        base_data = cls.get_base_data(base_fps, base_dset,
                                      base_gid, base_handler,
                                      daily_avg=daily_avg)

        bias_mean = bias_data.mean()
        bias_std = bias_data.std()
        base_mean = base_data.mean()
        base_std = base_data.std()

        scalar = (base_std / bias_std)
        adder = (base_mean - bias_mean * scalar)

        return scalar, adder

    def write_outputs(self, fp_out):
        """Write outputs to an .h5 file.

        Parameters
        ----------
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        """
        if fp_out is not None:
            with h5py.File(fp_out, 'a') as f:
                lat = self.bias_dh.lat_lon[..., 0]
                lon = self.bias_dh.lat_lon[..., 1]
                f.create_dataset('latitude', data=lat)
                f.create_dataset('longitude', data=lon)
                for name in self.bias_features:
                    f.create_dataset(f'{name}_scalar', data=self.scalar)
                    f.create_dataset(f'{name}_adder', data=self.adder)

                logger.info('Wrote scalar adder factors to file: {}'
                            .format(fp_out))

    def run(self, knn, threshold=0.6, fp_out=None, max_workers=None,
            daily_avg=True):
        """Run linear correction factor calculations for every site in the bias
        dataset

        Parameters
        ----------
        knn : int
            Number of nearest neighbors to aggregate from the base data when
            comparing to a single site from the bias data.
        threshold : float
            If the bias data coordinate is on average further from the base
            data coordinates than this threshold, no bias correction factors
            will be calculated directly and will just be filled from nearest
            neighbor.
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        max_workers : int
            Number of workers to run in parallel. 1 is serial and None is all
            available.
        daily_avg : bool
            Flag to do temporal daily averaging of the base data.

        Returns
        -------
        scalar : np.ndarray
            2D array of scalar factors corresponding to the bias raster data
            shape (lat, lon)
        adder : np.ndarray
            2D array of adder factors corresponding to the bias raster data
            shape (lat, lon)
        """
        logger.debug('Starting linear correction calculation...')

        self.scalar = np.full(self.bias_gid_raster.shape, np.nan, np.float32)
        self.adder = np.full(self.bias_gid_raster.shape, np.nan, np.float32)

        if max_workers == 1:
            logger.debug('Running serial calculation.')
            for i, (bias_gid, row) in enumerate(self.bias_meta.iterrows()):
                raster_loc = np.where(self.bias_gid_raster == bias_gid)
                coord = row[['latitude', 'longitude']]
                dist, base_gid = self.base_tree.query(coord, k=knn)

                if np.mean(dist) < threshold:
                    bias_data = self.get_bias_data(bias_gid)
                    out = self._run_single(bias_data, self.base_fps,
                                           self.base_dset, base_gid,
                                           self.base_handler, daily_avg)
                    self.scalar[raster_loc] = out[0]
                    self.adder[raster_loc] = out[1]

                logger.info('Completed bias calculations for {} out of {} '
                            'sites'.format(i + 1, len(self.bias_meta)))

        else:
            logger.debug('Running parallel calculation with {} workers.'
                         .format(max_workers))
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                futures = {}
                for bias_gid, bias_row in self.bias_meta.iterrows():
                    raster_loc = np.where(self.bias_gid_raster == bias_gid)
                    coord = bias_row[['latitude', 'longitude']]
                    dist, base_gid = self.base_tree.query(coord, k=knn)

                    if dist.mean() < threshold:
                        bias_data = self.get_bias_data(bias_gid)

                        future = exe.submit(self._run_single, bias_data,
                                            self.base_fps, self.base_dset,
                                            base_gid, self.base_handler,
                                            daily_avg)
                        futures[future] = raster_loc

                logger.debug('Finished launching futures.')
                for i, future in enumerate(as_completed(futures)):
                    raster_loc = futures[future]
                    self.scalar[raster_loc] = future.result()[0]
                    self.adder[raster_loc] = future.result()[1]

                    logger.info('Completed bias calculations for {} out of {} '
                                'sites'.format(i + 1, len(futures)))

        self.scalar = nn_fill_array(self.scalar)
        self.adder = nn_fill_array(self.adder)

        self.write_outputs(fp_out)

        return self.scalar, self.adder
