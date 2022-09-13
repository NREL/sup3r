# -*- coding: utf-8 -*-
"""Custom sup3r solar module. This primarily converts GAN output clearsky ratio
to GHI, DNI, and DHI using NSRDB data and utility modules like DISC

Note that clearsky_ratio is assumed to be clearsky ghi ratio and is calculated
as daily average GHI / daily average clearsky GHI.
"""
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from farms.disc import disc
from farms.utilities import calc_dhi, dark_night
from rex import Resource, MultiTimeResource


class Solar:
    """Custom sup3r solar module. This primarily converts GAN output clearsky
    ratio to GHI, DNI, and DHI using NSRDB data and utility modules like
    DISC"""

    def __init__(self, sup3r_fps, nsrdb_fp, tz=-6, agg_factor=1,
                 nn_threshold=0.5, cloud_threshold=0.99):
        """
        Parameters
        ----------
        sup3r_fps : str | list
            Full .h5 filepath(s) to one or more sup3r GAN output .h5 chunk
            files containing clearsky_ratio, time_index, and meta. These files
            must have the same meta data but can be sequential and ordered
            temporal chunks. The data in this file has been rolled by tz
            to a local time (assumed that the GAN was trained on local time
            solar data) and will be converted back into UTC, so it's wise to
            include some padding in the sup3r_fps file list.
        nsrdb_fp : str
            Filepath to NSRDB .h5 file containing clearsky_ghi, clearsky_dni,
            clearsky_dhi data.
        tz : int
            The timezone offset for the data in sup3r_fps. It is assumed that
            the GAN is trained on data in local time and therefore the output
            in sup3r_fps should be treated as local time. For example, -6 is
            CST which is default for CONUS training data.
        agg_factor : int
            Spatial aggregation factor for nsrdb-to-GAN-meta e.g. the number of
            NSRDB spatial pixels to average for a single sup3r GAN output site.
        nn_threshold : float
            The KDTree nearest neighbor threshold that determines how far the
            sup3r GAN output data has to be from the NSRDB source data to get
            irradiance=0. Note that is value is in decimal degrees which is a
            very approximate way to determine real distance.
        cloud_threshold : float
            Clearsky ratio threshold below which the data is considered cloudy
            and DNI is calculated using DISC.
        """

        self.agg_factor = agg_factor
        self.nn_threshold = nn_threshold
        self.cloud_threshold = cloud_threshold
        self.tz = tz
        self._nsrdb_fp = nsrdb_fp
        self._sup3r_fps = sup3r_fps
        if isinstance(self._sup3r_fps, str):
            self._sup3r_fps = [self._sup3r_fps]

        self.gan_data = MultiTimeResource(self._sup3r_fps)
        self.nsrdb = Resource(self._nsrdb_fp)

        # cached variables
        self._idnn = None
        self._dist = None
        self._sza = None
        self._cs_ratio = None
        self._ghi = None
        self._dni = None
        self._dhi = None

        self.preflight()

    def preflight(self):
        """Run preflight checks on source data to make sure everything will
        work together."""
        assert 'clearsky_ratio' in self.gan_data.dsets
        assert 'clearsky_ghi' in self.nsrdb.dsets
        assert 'clearsky_dni' in self.nsrdb.dsets
        assert 'solar_zenith_angle' in self.nsrdb.dsets
        assert 'surface_pressure' in self.nsrdb.dsets
        assert isinstance(self.nsrdb_tslice, slice)

        ti_gan = self.gan_data.time_index
        ti_gan_1 = np.roll(ti_gan, 1)
        delta = (ti_gan - ti_gan_1)[1:].mean().total_seconds()
        msg = ('Its assumed that the sup3r GAN output solar data will be '
               'hourly but received time index: {}'.format(ti_gan))
        assert delta == 3600, msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if type is not None:
            raise

    def close(self):
        """Close all internal file handlers"""
        self.gan_data.close()
        self.nsrdb.close()

    @property
    def idnn(self):
        """Get the nearest neighbor meta data indices from the NSRDB data that
        correspond to the sup3r GAN data

        Returns
        -------
        idnn : np.ndarray
            2D array of length (n_sup3r_sites, agg_factor) where the values are
            meta data indices from the NSRDB.
        """
        if self._idnn is None:
            tree = KDTree(self.nsrdb.meta[['latitude', 'longitude']])
            out = tree.query(self.gan_data.meta[['latitude', 'longitude']])
            self._dist, self._idnn = out
            if len(self._idnn.shape) == 1:
                self._dist = np.expand_dims(self._dist, axis=1)
                self._idnn = np.expand_dims(self._idnn, axis=1)
        return self._idnn

    @property
    def dist(self):
        """Get the nearest neighbor distances from the sup3r GAN data sites to
        the NSRDB nearest neighbors.

        Returns
        -------
        dist : np.ndarray
            2D array of length (n_sup3r_sites, agg_factor) where the values are
            decimal degree distances from the sup3r sites to the nsrdb nearest
            neighbors.
        """
        if self._dist is None:
            _ = self.idnn
        return self._dist

    @property
    def out_of_bounds(self):
        """Get a boolean mask for the sup3r data that is out of bounds (too far
        from the NSRDB data).

        Returns
        -------
        out_of_bounds : np.ndarray
            1D boolean array with length == number of sup3r GAN sites. True if
            the site is too far from the NSRDB.
        """
        return (self.dist > self.nn_threshold).any(axis=1)

    @property
    def nsrdb_tslice(self):
        """Get the time slice of the NSRDB data corresponding to the sup3r GAN
        output."""
        ti_nsrdb = self.nsrdb.time_index
        ti_gan = self.gan_data.time_index
        mask = ti_nsrdb.isin(ti_gan)
        msg = ('Time index intersection of the NSRDB time index and sup3r GAN '
               'output does not have more than 24 common timesteps! Something '
               'went wrong.\nNSRDB time index: \n{}\nSup3r GAN output time '
               'index:\n{}'.format(ti_nsrdb, ti_gan))
        assert mask.sum() > 24, msg
        ilocs = np.where(mask)[0]
        t0, t1 = ilocs[0], ilocs[-1] + 1
        step = pd.Series(np.diff(ilocs)).mode().values[0]
        return slice(t0, t1, step)

    @property
    def clearsky_ratio(self):
        """Get the clearsky ghi ratio data from the GAN output, rolled from tz
        to UTC.

        Returns
        -------
        clearsky_ratio : np.ndarray
            2D array with shape (time, sites) in UTC.
        """
        if self._cs_ratio is None:
            self._cs_ratio = self.gan_data['clearsky_ratio']
            self._cs_ratio = np.roll(self._cs_ratio, -self.tz, axis=0)

            # if tz is negative, roll to utc is positive, and the beginning of
            # the dataset is rolled over from the end and must be backfilled,
            # otherwise you can get seams
            self._cs_ratio[:-self.tz, :] = self._cs_ratio[-self.tz, :]

        return self._cs_ratio

    @property
    def sza(self):
        """Get the solar zenith angle (degrees)

        Returns
        -------
        sza : np.ndarray
            2D array with shape (time, sites) in UTC.
        """
        if self._sza is None:
            self._sza = self.get_nsrdb_data('solar_zenith_angle')
        return self._sza

    @property
    def ghi(self):
        """Get the ghi (W/m2) based on the GAN output clearsky ratio +
        clearsky_ghi in UTC.

        Returns
        -------
        ghi : np.ndarray
            2D array with shape (time, sites) in UTC.
        """
        if self._ghi is None:
            self._ghi = (self.get_nsrdb_data('clearsky_ghi')
                         * self.clearsky_ratio)
            self._ghi[:, self.out_of_bounds] = 0
        return self._ghi

    @property
    def dni(self):
        """Get the dni (W/m2) which is clearsky dni (from NSRDB) when the GAN
        output is clear (clearsky_ratio is > cloud_threshold), and calculated
        from the DISC model when cloudy

        Returns
        -------
        dni : np.ndarray
            2D array with shape (time, sites) in UTC.
        """
        if self._dni is None:
            self._dni = self.get_nsrdb_data('clearsky_dni')
            pressure = self.get_nsrdb_data('surface_pressure')
            doy = self.gan_data.time_index.day_of_year.values
            cloudy_dni = disc(self.ghi, self.sza, doy, pressure=pressure)
            cloudy_dni = np.minimum(self._dni, cloudy_dni)
            self._dni[self.cloud_mask] = cloudy_dni[self.cloud_mask]
            self._dni = dark_night(self._dni, self.sza)
            self._dni[:, self.out_of_bounds] = 0
        return self._dni

    @property
    def dhi(self):
        """Get the dhi (W/m2) which is calculated based on the simple
        relationship between GHI, DNI and solar zenith angle.

        Returns
        -------
        dhi : np.ndarray
            2D array with shape (time, sites) in UTC.
        """
        if self._dhi is None:
            self._dhi, self._dni = calc_dhi(self.dni, self.ghi, self.sza)
            self._dhi = dark_night(self._dhi, self.sza)
            self._dhi[:, self.out_of_bounds] = 0
        return self._dhi

    @property
    def cloud_mask(self):
        """Get the cloud mask (True if cloudy) based on the GAN output clearsky
        ratio in UTC.

        Returns
        -------
        cloud_mask : np.ndarray
            2D array with shape (time, sites) in UTC.
        """
        return self.clearsky_ratio < self.cloud_threshold

    def get_nsrdb_data(self, dset):
        """Get an NSRDB dataset with spatial index corresponding to the sup3r
        GAN output data, averaged across the agg_factor.

        Parameters
        ----------
        dset : str
            Name of dataset to retrieve from NSRDB source file

        Returns
        -------
        out : np.ndarray
            Dataset of shape (time, sites) where time and sites correspond to
            the same shape as the sup3r GAN output data and if agg_factor > 1
            the sites is an average across multiple NSRDB sites.
        """

        out = None

        for idx in range(self.idnn.shape[1]):
            temp = self.nsrdb[dset, self.nsrdb_tslice, self.idnn[:, idx]]
            temp = temp.astype(np.float32)

            if out is None:
                out = temp
            else:
                out += temp

        out /= self.idnn.shape[1]

        return out
