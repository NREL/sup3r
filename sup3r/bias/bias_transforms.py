# -*- coding: utf-8 -*-
"""Bias correction transformation functions."""
import logging
import os
from warnings import warn

import numpy as np
from rex import Resource
from scipy.ndimage.filters import gaussian_filter

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
        assert dset_scalar in res.dsets and dset_adder in res.dsets, msg
        scalar = res[dset_scalar, slice_y, slice_x]
        adder = res[dset_adder, slice_y, slice_x]
        return scalar, adder


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
