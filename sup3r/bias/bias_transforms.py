# -*- coding: utf-8 -*-
"""Bias correction transformation functions."""

import logging
import os
from warnings import warn

import numpy as np
import pandas as pd
from rex import Resource
from rex.utilities.bc_utils import QuantileDeltaMapping
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def _get_factors(lat_lon, ds, bias_fp, threshold=0.1):
    with Resource(bias_fp) as res:
        lat = np.expand_dims(res['latitude'], axis=-1)
        lon = np.expand_dims(res['longitude'], axis=-1)
        lat_lon_bc = np.dstack((lat, lon))
        diff = lat_lon_bc - lat_lon[:1, :1]
        diff = np.hypot(diff[..., 0], diff[..., 1])
        idy, idx = np.where(diff == diff.min())
        slice_y = slice(idy[0], idy[0] + lat_lon.shape[0])
        slice_x = slice(idx[0], idx[0] + lat_lon.shape[1])

        if diff.min() > threshold:
            msg = ('The DataHandler top left coordinate of {} '
                   'appears to be {} away from the nearest '
                   'bias correction coordinate of {} from {}. '
                   'Cannot apply bias correction.'.format(
                       lat_lon, diff.min(), lat_lon_bc[idy, idx],
                       os.path.basename(bias_fp),
                   ))
            logger.error(msg)
            raise RuntimeError(msg)

        res_names = [r.lower() for r in res.dsets]
        missing = [d for d in ds.values() if d.lower() not in res_names]
        msg = f'Missing {" and ".join(missing)} in resource: {bias_fp}.'
        assert missing == [], msg

        varnames = {k: res.dsets[res_names.index(ds[k].lower())] for k in ds}
        out = {k: res[varnames[k], slice_y, slice_x] for k in ds}

    return out


def get_spatial_bc_factors(lat_lon, feature_name, bias_fp, threshold=0.1):
    """Get bc factors (scalar/adder) for the given feature for the given
    domain (specified by lat_lon).

    Parameters
    ----------
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "{feature_name}_scalar" and "{feature_name}_adder" will be retrieved
        from bias_fp.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "{feature_name}_scalar" and "{feature_name}_adder" that are
        the full low-resolution shape of the forward pass input that will be
        sliced using lr_padded_slice for the current chunk.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error is
        raised.
    """
    ds = {'scalar': f'{feature_name}_scalar',
          'adder': f'{feature_name}_adder'}
    out = _get_factors(lat_lon, ds, bias_fp, threshold)

    return out['scalar'], out['adder']


def get_spatial_bc_quantiles(lat_lon: np.array,
                             base_dset: str,
                             feature_name: str,
                             bias_fp: str,
                             threshold: float = 0.1):
    """Statistical distributions previously estimated for given lat/lon points

    Recover the parameters that describe the statistical distribution
    previously estimated with
    :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for three
    datasets: ``base`` (historical reference), ``bias`` (historical biased
    reference), and ``bias_fut`` (the future biased dataset, usually the data
    to correct).

    Parameters
    ----------
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    base_dset : str
        Name of feature used as historical reference. A Dataset with name
        "base_{base_dset}_params" will be retrieved from ``bias_fp``.
    feature_name : str
        Name of the feature that is being corrected. Datasets with names
        "bias_{feature_name}_params" and "bias_fut_{feature_name}_params" will
        be retrieved from ``bias_fp``.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "base_{base_dset}_params", "bias_{feature_name}_params", and
        "bias_fut_{feature_name}_params" that define the statistical
        distributions.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error
        is raised.

    Returns
    -------
    base : np.array
        Parameters used to define the statistical distribution estimated for
        the ``base_dset``. It has a shape of (I, J, P), where (I, J) are the
        same first two dimensions of the given `lat_lon` and P is the number
        of parameters and depends on the type of distribution. See
        :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for more
        details.
    bias : np.array
        Parameters used to define the statistical distribution estimated for
        (historical) ``feature_name``. It has a shape of (I, J, P), where
        (I, J) are the same first two dimensions of the given `lat_lon` and P
        is the number of parameters and depends on the type of distribution.
        See :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for
        more details.
    bias_fut : np.array
        Parameters used to define the statistical distribution estimated for
        (future) ``feature_name``. It has a shape of (I, J, P), where (I, J)
        are the same first two dimensions of the given `lat_lon` and P is the
        number of parameters used and depends on the type of distribution. See
        :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for more
        details.
    cfg : dict
        Metadata used to guide how to use of the previous parameters on
        reconstructing the statistical distributions. For instance,
        `cfg['dist']` defines the type of distribution. See
        :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for more
        details, including which metadata is saved.

    Warnings
    --------
    Be careful selecting which `bias_fp` to use. In particular, if
    "bias_fut_{feature_name}_params" is representative for the desired target
    period.

    See Also
    --------
    sup3r.bias.qdm.QuantileDeltaMappingCorrection
        Estimate the statistical distributions loaded here.

    Examples
    --------
    >>> lat_lon = np.array([
    ...              [39.649033, -105.46875 ],
    ...              [39.649033, -104.765625]])
    >>> params, cfg = get_spatial_bc_quantiles(
    ...                 lat_lon, "ghi", "rsds", "./dist_params.hdf")
    """
    ds = {'base': f'base_{base_dset}_params',
          'bias': f'bias_{feature_name}_params',
          'bias_fut': f'bias_fut_{feature_name}_params',
          }
    params = _get_factors(lat_lon, ds, bias_fp, threshold)

    with Resource(bias_fp) as res:
        cfg = res.global_attrs

    return params, cfg


def global_linear_bc(input, scalar, adder, out_range=None):
    """Bias correct data using a simple global *scalar +adder method.

    Parameters
    ----------
    input : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    scalar : float
        Scalar (multiplicative) value to apply to input data.
    adder : float
        Adder value to apply to input data.
    out_range : None | tuple
        Option to set floor/ceiling values on the output data.

    Returns
    -------
    out : np.ndarray
        out = input * scalar + adder
    """
    out = input * scalar + adder
    if out_range is not None:
        out = np.maximum(out, np.min(out_range))
        out = np.minimum(out, np.max(out_range))
    return out


def local_linear_bc(input,
                    lat_lon,
                    feature_name,
                    bias_fp,
                    lr_padded_slice=None,
                    out_range=None,
                    smoothing=0,
                    ):
    """Bias correct data using a simple annual (or multi-year) *scalar +adder
    method on a site-by-site basis.

    Parameters
    ----------
    input : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "{feature_name}_scalar" and "{feature_name}_adder" will be retrieved
        from bias_fp.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "{feature_name}_scalar" and "{feature_name}_adder" that are
        the full low-resolution shape of the forward pass input that will be
        sliced using lr_padded_slice for the current chunk.
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included in the kwargs for the active chunk.
        If this is None, no slicing will be done and the full bias correction
        source shape will be used.
    out_range : None | tuple
        Option to set floor/ceiling values on the output data.
    smoothing : float
        Value to use to smooth the scalar/adder data. This can reduce the
        effect of extreme values within aggregations over large number of
        pixels.  This value is the standard deviation for the gaussian_filter
        kernel.

    Returns
    -------
    out : np.ndarray
        out = input * scalar + adder
    """

    scalar, adder = get_spatial_bc_factors(lat_lon, feature_name, bias_fp)
    # 3D bias correction factors have seasonal/monthly correction in last axis
    if len(scalar.shape) == 3 and len(adder.shape) == 3:
        scalar = scalar.mean(axis=-1)
        adder = adder.mean(axis=-1)

    if lr_padded_slice is not None:
        spatial_slice = (lr_padded_slice[0], lr_padded_slice[1])
        scalar = scalar[spatial_slice]
        adder = adder[spatial_slice]

    if np.isnan(scalar).any() or np.isnan(adder).any():
        msg = ('Bias correction scalar/adder values had NaNs for '
               f'"{feature_name}" from: {bias_fp}')
        logger.warning(msg)
        warn(msg)

    scalar = np.expand_dims(scalar, axis=-1)
    adder = np.expand_dims(adder, axis=-1)

    scalar = np.repeat(scalar, input.shape[-1], axis=-1)
    adder = np.repeat(adder, input.shape[-1], axis=-1)

    if smoothing > 0:
        for idt in range(scalar.shape[-1]):
            scalar[..., idt] = gaussian_filter(scalar[..., idt],
                                               smoothing,
                                               mode='nearest')
            adder[..., idt] = gaussian_filter(adder[..., idt],
                                              smoothing,
                                              mode='nearest')

    out = input * scalar + adder
    if out_range is not None:
        out = np.maximum(out, np.min(out_range))
        out = np.minimum(out, np.max(out_range))

    return out


def monthly_local_linear_bc(input,
                            lat_lon,
                            feature_name,
                            bias_fp,
                            time_index,
                            lr_padded_slice=None,
                            temporal_avg=True,
                            out_range=None,
                            smoothing=0,
                            ):
    """Bias correct data using a simple monthly *scalar +adder method on a
    site-by-site basis.

    Parameters
    ----------
    input : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "{feature_name}_scalar" and "{feature_name}_adder" will be retrieved
        from bias_fp.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "{feature_name}_scalar" and "{feature_name}_adder" that are
        the full low-resolution shape of the forward pass input that will be
        sliced using lr_padded_slice for the current chunk.
    time_index : pd.DatetimeIndex
        DatetimeIndex object associated with the input data temporal axis
        (assumed 3rd axis e.g. axis=2). Note that if this method is called as
        part of a sup3r resolution forward pass, the time_index will be
        included automatically for the current chunk.
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included automatically in the kwargs for the
        active chunk. If this is None, no slicing will be done and the full
        bias correction source shape will be used.
    temporal_avg : bool
        Take the average scalars and adders for the chunk's time index, this
        will smooth the transition of scalars/adders from month to month if
        processing small chunks. If processing the full annual time index, set
        this to False.
    out_range : None | tuple
        Option to set floor/ceiling values on the output data.
    smoothing : float
        Value to use to smooth the scalar/adder data. This can reduce the
        effect of extreme values within aggregations over large number of
        pixels.  This value is the standard deviation for the gaussian_filter
        kernel.

    Returns
    -------
    out : np.ndarray
        out = input * scalar + adder
    """
    scalar, adder = get_spatial_bc_factors(lat_lon, feature_name, bias_fp)

    assert len(scalar.shape) == 3, 'Monthly bias correct needs 3D scalars'
    assert len(adder.shape) == 3, 'Monthly bias correct needs 3D adders'

    if lr_padded_slice is not None:
        spatial_slice = (lr_padded_slice[0], lr_padded_slice[1])
        scalar = scalar[spatial_slice]
        adder = adder[spatial_slice]

    imonths = time_index.month.values - 1
    scalar = scalar[..., imonths]
    adder = adder[..., imonths]

    if temporal_avg:
        scalar = scalar.mean(axis=-1)
        adder = adder.mean(axis=-1)
        scalar = np.expand_dims(scalar, axis=-1)
        adder = np.expand_dims(adder, axis=-1)
        scalar = np.repeat(scalar, input.shape[-1], axis=-1)
        adder = np.repeat(adder, input.shape[-1], axis=-1)
        if len(time_index.month.unique()) > 2:
            msg = ('Bias correction method "monthly_local_linear_bc" was used '
                   'with temporal averaging over a time index with >2 months.')
            warn(msg)
            logger.warning(msg)

    if np.isnan(scalar).any() or np.isnan(adder).any():
        msg = ('Bias correction scalar/adder values had NaNs for '
               f'"{feature_name}" from: {bias_fp}')
        logger.warning(msg)
        warn(msg)

    if smoothing > 0:
        for idt in range(scalar.shape[-1]):
            scalar[..., idt] = gaussian_filter(scalar[..., idt],
                                               smoothing,
                                               mode='nearest')
            adder[..., idt] = gaussian_filter(adder[..., idt],
                                              smoothing,
                                              mode='nearest')

    out = input * scalar + adder
    if out_range is not None:
        out = np.maximum(out, np.min(out_range))
        out = np.minimum(out, np.max(out_range))

    return out


def local_qdm_bc(data: np.ndarray,
                 lat_lon: np.ndarray,
                 base_dset: str,
                 feature_name: str,
                 bias_fp,
                 time_index: pd.DatetimeIndex,
                 lr_padded_slice=None,
                 threshold=0.1,
                 relative=True,
                 no_trend=False,
                 ):
    """Bias correction using QDM

    Apply QDM to correct bias on the given data. It assumes that the required
    statistical distributions were previously estimated and saved in
    ``bias_fp``.

    Assume CDF for the nearest day of year (doy) is representative.

    Parameters
    ----------
    data : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    lat_lon : np.ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    base_dset :
        Name of feature that is used as (historical) reference. Dataset with
        names "base_{base_dset}_params" will be retrieved.
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "bias_{feature_name}_params" and "bias_fut_{feature_name}_params" will
        be retrieved.
    time_index : pd.DatetimeIndex
        DatetimeIndex object associated with the input data temporal axis
        (assumed 3rd axis e.g. axis=2). Note that if this method is called as
        part of a sup3r resolution forward pass, the time_index will be
        included automatically for the current chunk.
    bias_fp : str
        Filepath to statistical distributions file from the bias calc module.
        Must have datasets "bias_{feature_name}_params",
        "bias_fut_{feature_name}_params", and "base_{base_dset}_params" that
        are the parameters to define the statistical distributions to be used
        to correct the given `data`.
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included automatically in the kwargs for the
        active chunk. If this is None, no slicing will be done and the full
        bias correction source shape will be used.
    no_trend: bool, default=False
        An option to ignore the trend component of the correction, thus
        resulting in an ordinary Quantile Mapping, i.e. corrects the bias by
        comparing the distributions of the biased dataset with a reference
        datasets. See
        ``params_mf`` of :class:`rex.utilities.bc_utils.QuantileDeltaMapping`.
        Note that this assumes that params_mh is the data distribution
        representative for the target data.

    Returns
    -------
    out : np.ndarray
        The input data corrected by QDM. Its shape is the same of the input
        (spatial, spatial, time_window, temporal). The dimension time_window
        aligns with the number of time windows defined in a year, while
        temporal aligns with the time of the data.

    See Also
    --------
    sup3r.bias.qdm.QuantileDeltaMappingCorrection :
        Estimate probability distributions required by QDM method

    Notes
    -----
    Be careful selecting `bias_fp`. Usually, the input `data` used here would
    be related to the dataset used to estimate
    "bias_fut_{feature_name}_params".

    Keeping arguments as consistent as possible with `local_linear_bc()`, thus
    a 4D data (spatial, spatial, time_window, temporal), and lat_lon (n_lats,
    n_lons, [lat, lon]). But `QuantileDeltaMapping()`, from rex library,
    expects an array, (time, space), thus we need to re-organize our input to
    match that, and in the end bring it back to (spatial, spatial, time_window,
    temporal). This is still better than maintaining the same functionality
    consistent in two libraries.

    Also, :class:`rex.utilities.bc_utils.QuantileDeltaMapping` expects params
    to be 2D (space, N-params).

    See Also
    --------
    rex.utilities.bc_utils.QuantileDeltaMapping :
        Core QDM transformation.

    Examples
    --------
    >>> unbiased = local_qdm_bc(biased_array, lat_lon_array, "ghi", "rsds",
    ...                         "./dist_params.hdf")
    """
    # Confirm that the given time matches the expected data size
    assert (
        data.shape[2] == time_index.size
    ), 'Time should align with data 3rd dimension'

    base, bias, bias_fut, cfg = get_spatial_bc_quantiles(lat_lon,
                                                         base_dset,
                                                         feature_name,
                                                         bias_fp,
                                                         threshold)

    if lr_padded_slice is not None:
        spatial_slice = (lr_padded_slice[0], lr_padded_slice[1])
        base = base[spatial_slice]
        bias = bias[spatial_slice]
        bias_fut = bias_fut[spatial_slice]

    output = np.full_like(data, np.nan)
    nearest_window_idx = [
        np.argmin(abs(d - cfg['time_window_center']))
        for d in time_index.day_of_year
    ]
    for window_idx in set(nearest_window_idx):
        # Naming following the paper: observed historical
        oh = base[:, :, window_idx]
        # Modeled historical
        mh = bias[:, :, window_idx]
        # Modeled future
        mf = bias_fut[:, :, window_idx]

        # This satisfies the rex's QDM design
        if no_trend:
            mf = None
        else:
            mf = mf.reshape(-1, mf.shape[-1])
        # The distributions at this point, after selected the respective
        # time window with `window_idx`, are 3D (space, space, N-params)
        # Collapse 3D (space, space, N) into 2D (space**2, N)
        QDM = QuantileDeltaMapping(oh.reshape(-1, oh.shape[-1]),
                                   mh.reshape(-1, mh.shape[-1]),
                                   mf,
                                   dist=cfg['dist'],
                                   relative=relative,
                                   sampling=cfg['sampling'],
                                   log_base=cfg['log_base'],
                                   )

        subset_idx = nearest_window_idx == window_idx
        subset = data[:, :, subset_idx]
        # input 3D shape (spatial, spatial, temporal)
        # QDM expects input arr with shape (time, space)
        tmp = subset.reshape(-1, subset.shape[-1]).T
        # Apply QDM correction
        tmp = QDM(tmp)
        # Reorgnize array back from  (time, space)
        # to (spatial, spatial, temporal)
        tmp = tmp.T.reshape(subset.shape)
        # Position output respecting original time axis sequence
        output[:, :, subset_idx] = tmp

    return output


def apply_zero_precipitation_rate(arr: np.ndarray, rate):
    """Enforce the zero precipitation rate
    Replace lowest values by zero to satisfy the given rate of zero
    precipitation.

    Parameters
    ----------
    arr : np.array
        An array of values to be analyzed. Usually precipitation but it
    rate : float or np.array
        Rate of zero, or negligible, days of precipitation.

    Returns
    -------
    corrected : np.array
        A copy of given array that satisfies the rate of zero precipitation
        days, i.e. the lowest values of precipitation are changed to zero
        to satisfy that rate.

    Examples
    --------
    >>> data = np.array([5, 0.1, np.nan, 0.2, 1])
    >>> apply_zero_precipitation_rate(data, 0.30)
    array([5. , 0. , nan, 0.2, 1. ])
    """
    assert arr.ndim == 3
    assert rate.ndim >= 2
    assert arr.shape[:2] == rate.shape[:2]

    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            r = rate[i, j, 0]
            if np.isfinite(r):
                a = arr[i, j]
                valid = np.isfinite(a)
                idx = np.argsort(a)[valid][:round(r * valid.astype('i').sum())]
                a[idx] = 0
                arr[i, j] = a

    return arr


def get_spatial_bc_presrat(lat_lon: np.array,
                           base_dset: str,
                           feature_name: str,
                           bias_fp: str,
                           threshold: float = 0.1):
    """Statistical distributions previously estimated for given lat/lon points

    Recover the parameters that describe the statistical distribution
    previously estimated with
    :class:`~sup3r.bias.bias_calc.QuantileDeltaMappingCorrection` for three
    datasets: ``base`` (historical reference), ``bias`` (historical biased
    reference), and ``bias_fut`` (the future biased dataset, usually the data
    to correct).

    Parameters
    ----------
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    base_dset : str
        Name of feature used as historical reference. A Dataset with name
        "base_{base_dset}_params" will be retrieved from ``bias_fp``.
    feature_name : str
        Name of the feature that is being corrected. Datasets with names
        "bias_{feature_name}_params" and "bias_fut_{feature_name}_params" will
        be retrieved from ``bias_fp``.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "base_{base_dset}_params", "bias_{feature_name}_params", and
        "bias_fut_{feature_name}_params" that define the statistical
        distributions.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error
        is raised.

    Returns
    -------
    base : np.array
        Parameters used to define the statistical distribution estimated for
        the ``base_dset``. It has a shape of (I, J, P), where (I, J) are the
        same first two dimensions of the given `lat_lon` and P is the number
        of parameters and depends on the type of distribution. See
        :class:`~sup3r.bias.bias_calc.QuantileDeltaMappingCorrection` for more
        details.
    bias : np.array
        Parameters used to define the statistical distribution estimated for
        (historical) ``feature_name``. It has a shape of (I, J, P), where
        (I, J) are the same first two dimensions of the given `lat_lon` and P
        is the number of parameters and depends on the type of distribution.
        See :class:`~sup3r.bias.bias_calc.QuantileDeltaMappingCorrection` for
        more details.
    bias_fut : np.array
        Parameters used to define the statistical distribution estimated for
        (future) ``feature_name``. It has a shape of (I, J, P), where (I, J)
        are the same first two dimensions of the given `lat_lon` and P is the
        number of parameters used and depends on the type of distribution. See
        :class:`~sup3r.bias.bias_calc.QuantileDeltaMappingCorrection` for more
        details.
    cfg : dict
        Metadata used to guide how to use of the previous parameters on
        reconstructing the statistical distributions. For instance,
        `cfg['dist']` defines the type of distribution. See
        :class:`~sup3r.bias.bias_calc.QuantileDeltaMappingCorrection` for more
        details, including which metadata is saved.

    Warnings
    --------
    Be careful selecting which `bias_fp` to use. In particular, if
    "bias_fut_{feature_name}_params" is representative for the desired target
    period.

    See Also
    --------
    sup3r.bias.bias_calc.QuantileDeltaMappingCorrection
        Estimate the statistical distributions loaded here.

    Examples
    --------
    >>> lat_lon = np.array([
    ...              [39.649033, -105.46875 ],
    ...              [39.649033, -104.765625]])
    >>> params, cfg = get_spatial_bc_quantiles(
    ...                 lat_lon, "ghi", "rsds", "./dist_params.hdf")
    """
    ds = {'base': f'base_{base_dset}_params',
          'base_zero_rate': f'{base_dset}_zero_rate',
          'bias': f'bias_{feature_name}_params',
          'bias_fut': f'bias_fut_{feature_name}_params',
          'bias_mean_mh': f'{feature_name}_mean_mh',
          'bias_mean_mf': f'{feature_name}_mean_mf',
          }
    params = _get_factors(lat_lon, ds, bias_fp, threshold)

    with Resource(bias_fp) as res:
        cfg = res.global_attrs

    return params, cfg


def local_presrat_bc(data: np.ndarray,
                     time_index: np.ndarray,
                     lat_lon: np.ndarray,
                     base_dset: str,
                     feature_name: str,
                     bias_fp,
                     lr_padded_slice=None,
                     threshold=0.1,
                     relative=True,
                     no_trend=False):
    """Bias correction using PresRat

    Parameters
    ----------
    data : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    base_dset :
        Name of feature that is used as (historical) reference. Dataset with
        names "base_{base_dset}_params" will be retrieved.
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "bias_{feature_name}_params" and "bias_fut_{feature_name}_params" will
        be retrieved.
    bias_fp : str
        Filepath to statistical distributions file from the bias calc module.
        Must have datasets "bias_{feature_name}_params",
        "bias_fut_{feature_name}_params", and "base_{base_dset}_params" that
        are the parameters to define the statistical distributions to be used
        to correct the given `data`.
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included automatically in the kwargs for the
        active chunk. If this is None, no slicing will be done and the full
        bias correction source shape will be used.
    no_trend: bool, default=False
        An option to ignore the trend component of the correction, thus
        resulting in an ordinary Quantile Mapping, i.e. corrects the bias by
        comparing the distributions of the biased dataset with a reference
        datasets. See
        ``params_mf`` of :class:`rex.utilities.bc_utils.QuantileDeltaMapping`.
        Note that this assumes that params_mh is the data distribution
        representative for the target data.
    """
    assert data.ndim, "data was expected to be a 3D array"
    assert data.shape[-1] == time_index.size, \
        "The last dimension of data should be time"

    params, cfg = get_spatial_bc_presrat(lat_lon,
                                         base_dset,
                                         feature_name,
                                         bias_fp,
                                         threshold)
    base = params["base"]
    bias = params["bias"]
    bias_fut = params["bias_fut"]

    if lr_padded_slice is not None:
        spatial_slice = (lr_padded_slice[0], lr_padded_slice[1])
        base = base[spatial_slice]
        bias = bias[spatial_slice]
        bias_fut = bias_fut[spatial_slice]

    if no_trend:
        mf = None
    else:
        mf = bias_fut.reshape(-1, bias_fut.shape[-1])
    # The distributions are 3D (space, space, N-params)
    # Collapse 3D (space, space, N) into 2D (space**2, N)
    QDM = QuantileDeltaMapping(base.reshape(-1, base.shape[-1]),
                               bias.reshape(-1, bias.shape[-1]),
                               mf,
                               dist=cfg['dist'],
                               relative=relative,
                               sampling=cfg["sampling"],
                               log_base=cfg["log_base"])

    # input 3D shape (spatial, spatial, temporal)
    # QDM expects input arr with shape (time, space)
    tmp = data.reshape(-1, data.shape[-1]).T
    # Apply QDM correction
    tmp = QDM(tmp)
    # Reorgnize array back from  (time, space) to (spatial, spatial, temporal)
    tmp = tmp.T.reshape(data.shape)

    tmp = apply_zero_precipitation_rate(tmp, params["base_zero_rate"])

    month = time_index.month
    for m in range(12):
        idx = month == m + 1
        x_hat = tmp[:, :, idx].mean(axis=-1)
        k = params["bias_mean_mf"][:, :, m] / x_hat
        tmp[:, :, idx] *= k[:, :, np.newaxis]

    return tmp
