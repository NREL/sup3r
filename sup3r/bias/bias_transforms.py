# -*- coding: utf-8 -*-
"""Bias correction transformation functions."""
import logging
import os
from warnings import warn

import numpy as np
from rex import Resource
from rex.utilities.bc_utils import QuantileDeltaMapping
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


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
    dset_scalar = f'{feature_name}_scalar'
    dset_adder = f'{feature_name}_adder'
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

        msg = (f'Either {dset_scalar} or {dset_adder} not found in {bias_fp}.')
        dsets = [dset.lower() for dset in res.dsets]
        check = dset_scalar.lower() in dsets and dset_adder.lower() in dsets
        assert check, msg
        dset_scalar = res.dsets[dsets.index(dset_scalar.lower())]
        dset_adder = res.dsets[dsets.index(dset_adder.lower())]
        scalar = res[dset_scalar, slice_y, slice_x]
        adder = res[dset_adder, slice_y, slice_x]
    return scalar, adder


def get_spatial_bc_quantiles(lat_lon: np.array,
                             base_dset: str,
                             feature_name: str,
                             bias_fp: str,
                             threshold: float = 0.1
                             ):
    dset_base = f'base_{base_dset}_params'
    dset_bias = f'bias_{feature_name}_params'
    dset_bias_fut = f'bias_fut_{feature_name}_params'
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

        msg = (f'Either {dset_base} or {dset_bias} or {dset_bias_fut} not found in {bias_fp}.')
        dsets = [dset.lower() for dset in res.dsets]
        check = dset_base.lower() in dsets \
                and dset_bias.lower() in dsets \
                and dset_bias_fut.lower() in dsets
        assert check, msg
        dset_base = res.dsets[dsets.index(dset_base.lower())]
        dset_bias = res.dsets[dsets.index(dset_bias.lower())]
        dset_bias_fut = res.dsets[dsets.index(dset_bias_fut.lower())]

        base = res[dset_base, slice_y, slice_x]
        bias = res[dset_bias, slice_y, slice_x]
        bias_fut = res[dset_bias_fut, slice_y, slice_x]

        cfg = {k:v for k,v in res.h5.attrs.items() if k in ("dist", "sampling", "log_base")}

    return base, bias, bias_fut, cfg


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
                    lr_padded_slice,
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
                            lr_padded_slice,
                            time_index,
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
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included automatically in the kwargs for the
        active chunk. If this is None, no slicing will be done and the full
        bias correction source shape will be used.
    time_index : pd.DatetimeIndex
        DatetimeIndex object associated with the input data temporal axis
        (assumed 3rd axis e.g. axis=2). Note that if this method is called as
        part of a sup3r resolution forward pass, the time_index will be
        included automatically for the current chunk.
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


def local_qdm_bc(data: np.array,
                 lat_lon: np.array,
                 base_dset: str,
                 feature_name: str,
                 bias_fp,
                 relative=True):
    """Bias correction using QDM

    Apply QDM to correct bias on the given data. It assumes that the required
    statistical distributions were previously estimated and saved in
    `bias_fp`.

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

    Returns
    -------
    out : np.ndarray
        The input data corrected by QDM. Its shape is the same of the input
        (spatial, spatial, temporal)

    See Also
    --------
    sup3r.bias.bias_calc.QuantileDeltaMappingCorrection :
        Estimate probability distributions required by QDM method

    Notes
    -----
    Be careful selecting `bias_fp`. Usually, the input `data` used here would
    be related to the dataset used to estimate "bias_fut_{feature_name}_params".

    Keeping arguments consistent with `local_linear_bc()`, thus a 3D data
    (spatial, spatial, temporal), and lat_lon (n_lats, n_lons, [lat, lon]).
    But `QuantileDeltaMapping()`, from rex library, operates an array,
    (time, space), thus we need to re-organize our input to match that,
    and in the end bring it back to (spatial, spatial, temporal). This is
    still better than maintaining the same functionality consistent in two
    libraries.

    Also, rex's `QuantileDeltaMapping()` expects params to be 2D
    (space, N-params).

    See Also
    --------
    rex.utilities.bc_utils.QuantileDeltaMapping :
        Core QDM transformation.


    """
    base, bias, bias_fut, cfg = get_spatial_bc_quantiles(lat_lon,
                                                         base_dset,
                                                         feature_name,
                                                         bias_fp)

    # distributions are 3D (space, space, N-params)
    projection = lambda x: x.reshape(-1, x.shape[-1])
    # params expected to be 2D arrays (space, N-params)
    QDM = QuantileDeltaMapping(projection(base),
                               projection(bias),
                               projection(bias_fut),
                               dist=cfg["dist"],
                               relative=relative,
                               sampling=cfg["sampling"],
                               log_base=cfg["log_base"])

    # input 3D shape (spatial, spatial, temporal)
    # QDM expects input arr with shape (time, space)
    tmp = data.reshape(-1, data.shape[-1]).T
    # Apply QDM correction
    tmp = QDM(tmp)
    # Reorgnize array back from  (time, space) to (spatial, spatial, temporal)
    return tmp.T.reshape(data.shape)
