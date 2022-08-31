"""Utilities for stitching south east asia domains"""
# -*- coding: utf-8 -*-
# pylint: disable=all
import xarray as xr
import numpy as np
import logging
import xesmf as xe
import glob
import os

logger = logging.getLogger(__name__)


IN_FILENAME = '/projects/seasiawind/modeling/seasia/production/wrfouts'
IN_FILENAME += '/{year}/{month}/custom_wrfout_d0{domain}_{year}-{month}-*'
OUT_FILENAME = '/scratch/bbenton/seasiawind/stitched/'
OUT_FILENAME += '{year}/{month}/'


class Regridder:
    """Regridder class for stitching domains"""

    def __init__(self, lats, lons, min_lat, max_lat, min_lon, max_lon,
                 n_lats, n_lons):
        """
        Parameters
        ----------
        lats : ndarray
            Array of latitudes for input grid
        lons : ndarray
            Array of longitudes for input grid
        min_lat : float
            Minimum lat for output grid
        max_lat : float
            Maximum lat for output grid
        min_lon : float
            Minimum lon for output grid
        max_lon : float
            Maximum lon for output grid
        n_lats : int
            Number of lats for output grid
        n_lons : int
            Number of lons for output grid
        """
        self._regridder = None
        self.grid_in = {'lat': lats, 'lon': lons}
        lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, n_lons),
                                 np.linspace(min_lat, max_lat, n_lats))
        self.grid_out = {'lat': lats, 'lon': lons}
        self.new_lat_lon = np.zeros((*lats.shape, 2))
        self.new_lat_lon[..., 0] = lats
        self.new_lat_lon[..., 1] = lons

    @property
    def regridder(self):
        """Get regridder for grid_in to grid_out"""
        if self._regridder is None:
            self._regridder = xe.Regridder(self.grid_in, self.grid_out,
                                           "bilinear")
        return self._regridder

    def regrid_data(self, data_in):
        """Regrid data to output grid

        Parameters
        ----------
        data_in : xarray.Dataset
            input data handle

        Returns
        -------
        data_out : xarray.Dataset
            output data handle
        """
        times = data_in.Times.values
        data_out = self.regridder(data_in)
        data_out = data_out.rename({'lat': 'XLAT', 'lon': 'XLONG'})
        data_out = data_out.rename({'x': 'west_east', 'y': 'south_north'})
        data_out['Times'] = ('Time', times)
        data_out['XLAT'] = (('Time', 'south_north', 'west_east'),
                            np.repeat(np.expand_dims(data_out['XLAT'].values,
                                                     axis=0),
                                      len(times), axis=0))
        data_out['XLONG'] = (('Time', 'south_north', 'west_east'),
                             np.repeat(np.expand_dims(data_out['XLONG'].values,
                                                      axis=0),
                                       len(times), axis=0))
        return data_out


def get_files(year, month, input_pattern=IN_FILENAME,
              output_pattern=OUT_FILENAME):
    """Get input files for all domains to stitch together, and output file
    name

    Parameters
    ----------
    year : int
        Year for input files
    month : int
        Month for input files
    input_pattern : str
        Pattern for input files. Assumes pattern contains {month}, {year}, and
        {domain}
    output_pattern : str
        Pattern for output files
    """
    in_pattern = [input_pattern.format(year=year, month=str(month).zfill(2),
                                       domain=i) for i in range(1, 5)]
    input_files = {i + 1: sorted(glob.glob(in_pattern[i])) for i in range(4)}
    out_pattern = output_pattern.format(year=year, month=str(month).zfill(2))
    out_files = [os.path.join(out_pattern,
                              os.path.basename(input_files[1][i]).replace(
                                  'custom_wrfout_d01', 'stitched_wrfout'))
                 for i in range(len(input_files[1]))]
    return input_files, out_files


def get_handles(input_files):
    """Get handles for all domains. Keep needed fields

    Parameters
    ----------
    input_files : list
        List of input files for each domain. First file needs to be the file
        for the largest domain.

    Returns
    -------
    handles : list
        List of xarray.Dataset objects for each domain
    """
    handles = []
    for f in input_files:
        handle = xr.open_dataset(f)
        handle = handle[['U', 'V', 'PHB', 'PH', 'HGT', 'P', 'T', 'Times']]
        handles.append(handle)
    return handles


def unstagger_vars(handle):
    """Unstagger wind components

    Parameters
    ----------
    handle : xarray.Dataset
        Handle for a given domain

    Returns
    -------
    handle : xarray.Dataset
        Handle for a given domain, with unstaggered fields
    """

    handle['U'] = (('Time', 'bottom_top', 'south_north', 'west_east'),
                   np.apply_along_axis(forward_avg, 3, handle['U']))
    handle['V'] = (('Time', 'bottom_top', 'south_north', 'west_east'),
                   np.apply_along_axis(forward_avg, 2, handle['V']))
    handle['PHB'] = (('Time', 'bottom_top', 'south_north', 'west_east'),
                     np.apply_along_axis(forward_avg, 1, handle['PHB']))
    handle['PH'] = (('Time', 'bottom_top', 'south_north', 'west_east'),
                    np.apply_along_axis(forward_avg, 1, handle['PH']))
    return handle


def prune_levels(handle):
    """Prune pressure levels to reduce memory footprint

    Parameters
    ----------
    handle : xarray.Dataset
        Handle for a given domain

    Returns
    -------
    handle : xarray.Dataset
        Handle for a given domain, with reduced number of pressure levels
    """
    handle = handle.loc[dict(bottom_top=slice(0, 10))]
    return handle


def regrid_domain1(handles):
    """Regrid largest domain

    Parameters
    ----------
    handles : list
        List of xarray.Dataset objects for each domain

    Returns
    -------
    handles : list
        List of xarray.Dataset objects for each domain, with unstaggered
        variables and pruned pressure levels.
    """
    min_lat = np.min(handles[0].XLAT)
    min_lon = np.min(handles[0].XLONG)
    max_lat = np.max(handles[0].XLAT)
    max_lon = np.max(handles[0].XLONG)
    n_lons = handles[0].XLAT.shape[-1]
    n_lats = handles[0].XLAT.shape[1]
    d01_regridder = Regridder(handles[0].XLAT[0], handles[0].XLONG[0],
                              min_lat, max_lat, min_lon, max_lon,
                              3 * n_lats, 3 * n_lons)
    for i, _ in enumerate(handles):
        handles[i] = unstagger_vars(handles[i])
        handles[i] = prune_levels(handles[i])
    handles[0] = d01_regridder.regrid_data(handles[0])
    return handles


def forward_avg(array_in):
    """Forward average for use in unstaggering"""
    return (array_in[:-1] + array_in[1:]) * 0.5


def blend_domains(arr1, arr2, overlap=15):
    """Blend smaller domain edges

    Parameters
    ----------
    arr1 : ndarray
        Data array for largest domain
    arr2 : ndarray
        Data array for nested domain to stitch into larger domain

    Returns
    -------
    out : ndarray
        Data array with smaller domain blended into larger domain
    """
    out = arr2.copy()
    for i in range(overlap):
        alpha = i / overlap
        out[..., i, :] = out[..., i, :] * alpha + arr1[..., i, :] * (1 - alpha)
        out[..., -i, :] = (out[..., -i, :] * alpha
                           + arr1[..., -i, :] * (1 - alpha))
        out[..., :, i] = out[..., :, i] * alpha + arr1[..., :, i] * (1 - alpha)
        out[..., :, -i] = (out[..., :, -i] * alpha
                           + arr1[..., :, -i] * (1 - alpha))
    return out


def get_domain_region(handles, domain_num):
    """Get range for smaller domain

    Parameters
    ----------
    handles : list
        List of xarray.Dataset objects for each domain
    domain_num : int
        Domain number to get grid range for

    Returns
    -------
    lat_range : slice
        Slice corresponding to lat range of smaller domain within larger domain
    lon_range : slice
        Slice corresponding to lon range of smaller domain within larger domain
    min_lat : float
        Minimum lat for smaller domain
    max_lat : float
        Maximum lat for smaller domain
    min_lon : float
        Minimum lon for smaller domain
    max_lon : float
        Maximum lon for smaller domain
    n_lats : int
        Number of lats for smaller domain
    n_lons : int
        Number of lons for smaller domain
    """
    lats = handles[0].XLAT[0, :, 0]
    lons = handles[0].XLONG[0, 0, :]
    min_lat = np.min(handles[domain_num - 1].XLAT.values)
    min_lon = np.min(handles[domain_num - 1].XLONG.values)
    max_lat = np.max(handles[domain_num - 1].XLAT.values)
    max_lon = np.max(handles[domain_num - 1].XLONG.values)
    lat_mask = (min_lat <= lats) & (lats <= max_lat)
    lon_mask = (min_lon <= lons) & (lons <= max_lon)
    lat_idx = np.arange(len(lats))
    lon_idx = np.arange(len(lons))
    lat_range = slice(lat_idx[lat_mask][0], lat_idx[lat_mask][-1] + 1)
    lon_range = slice(lon_idx[lon_mask][0], lon_idx[lon_mask][-1] + 1)
    n_lats = len(lat_idx[lat_mask])
    n_lons = len(lon_idx[lon_mask])
    return (lat_range, lon_range, min_lat, max_lat, min_lon, max_lon,
            n_lats, n_lons)


def impute_domain(handles, domain_num):
    """Impute smaller domain in largest domain

    Parameters
    ----------
    handles : list
        List of xarray.Dataset objects for each domain
    domain_num : int
        Domain number to stitch into largest domain

    Returns
    -------
    handles : list
        List of xarray.Dataset objects for each domain
    """
    out = get_domain_region(handles, domain_num)
    (lat_range, lon_range, min_lat, max_lat, min_lon,
     max_lon, n_lats, n_lons) = out
    regridder = Regridder(handles[domain_num - 1].XLAT[0],
                          handles[domain_num - 1].XLONG[0],
                          min_lat, max_lat, min_lon, max_lon, n_lats, n_lons)
    handles[domain_num - 1] = regridder.regrid_data(handles[domain_num - 1])
    for field in handles[0]:
        if field not in ['Times']:
            arr1 = handles[0][field].loc[dict(south_north=lat_range,
                                              west_east=lon_range)]
            arr2 = handles[domain_num - 1][field]
            out = blend_domains(arr1, arr2)
            handles[0][field].loc[dict(south_north=lat_range,
                                       west_east=lon_range)] = out
    return handles


def stitch_and_save(year, month, input_pattern=IN_FILENAME,
                    output_pattern=OUT_FILENAME):
    """Stitch all smaller domains into largest domain and save output

    Parameters
    ----------
    year : int
        Year for input files
    month : int
        Month for input files
    input_pattern : str
        Pattern for input files. Assumes pattern contains {month}, {year}, and
        {domain}
    output_pattern : str
        Pattern for output files
    """
    logger.info(f'Getting file patterns for year={year}, month={month}')
    input_files, out_files = get_files(year, month, input_pattern,
                                       output_pattern)
    for i, out_file in enumerate(out_files):
        if not os.path.exists(out_file):
            logger.info(f'Getting domain files for year={year}, month={month},'
                        f' timestep={i}.')
            step_files = [input_files[d][i] for d in range(1, 5)]
            logger.info(f'Getting data handles for files: {step_files}')
            handles = get_handles(step_files)
            logger.info(f'Regridding domain 1 for year={year}, month={month}, '
                        f'timestep={i}')
            handles = regrid_domain1(handles)
            for j in range(2, 5):
                logger.info(f'Imputing domain {j} for year={year}, '
                            f'month={month}, timestep={i}')
                handles = impute_domain(handles, j)
            basedir = os.path.dirname(out_file)
            os.makedirs(basedir, exist_ok=True)
            handles[0].to_netcdf(out_file)
            logger.info(f'Saved stitched file to {out_file}')
