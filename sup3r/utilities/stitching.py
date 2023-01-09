"""Utilities for stitching south east asia domains

Example Use:

    stitch_and_save(year=2017, month=1,
                    input_pattern="wrfout_d0{domain}_{year}-{month}*",
                    out_pattern="{year}/{month}/", overlap=15, n_domains=4,
                    max_levels=10)

    This will combine 4 domains, for Jan 2017, using an overlap of 15 grid
    points the blend the domain edges, and save only the first 10 pressure
    levels. The stitched files will be saved in the directory specified with
    out_pattern.
"""
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import logging
from importlib import import_module
import glob
import os

logger = logging.getLogger(__name__)

SAVE_FEATURES = ['U', 'V', 'PHB', 'PH', 'HGT', 'P', 'PB', 'T', 'Times']


class Regridder:
    """Regridder class for stitching domains"""

    DEPENDENCIES = ['xesmf']

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
        self.check_dependencies()
        import xesmf as xe
        self.grid_in = {'lat': lats, 'lon': lons}
        lons, lats = np.meshgrid(np.linspace(min_lon, max_lon, n_lons),
                                 np.linspace(min_lat, max_lat, n_lats))
        self.grid_out = {'lat': lats, 'lon': lons}
        self.new_lat_lon = np.zeros((*lats.shape, 2))
        self.new_lat_lon[..., 0] = lats
        self.new_lat_lon[..., 1] = lons
        self.regridder = xe.Regridder(self.grid_in, self.grid_out,
                                      method='bilinear')

    @classmethod
    def check_dependencies(cls):
        """Check special dependencies for stitching module"""

        missing = []
        for name in cls.DEPENDENCIES:
            try:
                import_module(name)
            except ModuleNotFoundError:
                missing.append(name)

        if any(missing):
            msg = ('The sup3r stitching module depends on the following '
                   'special dependencies that were not found in the active '
                   'environment: {}'.format(missing))
            logger.error(msg)
            raise ModuleNotFoundError(msg)

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


def get_files(year, month, input_pattern, out_pattern, n_domains=4):
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
    out_pattern : str
        Pattern for output files. Assumes pattern contains {month} and {year}
    n_domains : int
        Number of domains to stitch together

    Returns
    -------
    input_files : dict
        Dictionary of input files with keys corresponding to domain number
    out_files : list
        List of output file names for final stitched output
    """
    in_pattern = [input_pattern.format(year=year, month=str(month).zfill(2),
                                       domain=i)
                  for i in range(1, n_domains + 1)]
    input_files = {i: sorted(glob.glob(in_pattern[i]))
                   for i in range(n_domains)}
    out_pattern = out_pattern.format(year=year, month=str(month).zfill(2))
    out_files = [os.path.join(out_pattern,
                              os.path.basename(input_files[0][i]).replace(
                                  'custom_wrfout_d01', 'stitched_wrfout'))
                 for i in range(len(input_files[0]))]
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
        logger.info(f'Getting handle for {f}')
        handle = xr.open_dataset(f)
        handle = handle[SAVE_FEATURES]
        handles.append(handle)
    return handles


def unstagger_vars(handles):
    """Unstagger variables for all handles

    Parameters
    ----------
    handles : list
        List of xarray.Dataset objects for each domain

    Returns
    -------
    handles : list
        List of xarray.Dataset objects for each domain, with unstaggered
        variables.
    """
    dims = ('Time', 'bottom_top', 'south_north', 'west_east')
    for i, handle in enumerate(handles):
        handles[i]['U'] = (dims, np.apply_along_axis(forward_avg, 3,
                                                     handle['U']))
        handles[i]['V'] = (dims, np.apply_along_axis(forward_avg, 2,
                                                     handle['V']))
        handles[i]['PHB'] = (dims, np.apply_along_axis(forward_avg, 1,
                                                       handle['PHB']))
        handles[i]['PH'] = (dims, np.apply_along_axis(forward_avg, 1,
                                                      handle['PH']))
    return handles


def prune_levels(handles, max_level=15):
    """Prune pressure levels to reduce memory footprint

    Parameters
    ----------
    handles : list
        List of xarray.Dataset objects for each domain
    max_level : int
        Max pressure level index

    Returns
    -------
    handles : list
        List of xarray.Dataset objects for each domain, with pruned pressure
        levels.
    """
    for i, handle in enumerate(handles):
        handles[i] = handle.loc[dict(bottom_top=slice(0, max_level))]
    return handles


def regrid_main_domain(handles):
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
    main_regridder = Regridder(handles[0].XLAT[0], handles[0].XLONG[0],
                               min_lat, max_lat, min_lon, max_lon,
                               3 * n_lats, 3 * n_lons)
    handles[0] = main_regridder.regrid_data(handles[0])
    return handles


def forward_avg(array_in):
    """Forward average for use in unstaggering"""
    return (array_in[:-1] + array_in[1:]) * 0.5


def blend_domains(arr1, arr2, overlap=50):
    """Blend smaller domain edges

    Parameters
    ----------
    arr1 : ndarray
        Data array for largest domain
    arr2 : ndarray
        Data array for nested domain to stitch into larger domain
    overlap : int
        Number of grid points to use for blending edges

    Returns
    -------
    out : ndarray
        Data array with smaller domain blended into larger domain
    """
    out = arr2.copy()
    for i in range(overlap):
        alpha = i / overlap
        beta = 1 - alpha
        out[..., i, :] = out[..., i, :] * alpha + arr1[..., i, :] * beta
        out[..., -i, :] = out[..., -i, :] * alpha + arr1[..., -i, :] * beta
        out[..., :, i] = out[..., :, i] * alpha + arr1[..., :, i] * beta
        out[..., :, -i] = out[..., :, -i] * alpha + arr1[..., :, -i] * beta
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
    min_lat = np.min(handles[domain_num].XLAT.values)
    min_lon = np.min(handles[domain_num].XLONG.values)
    max_lat = np.max(handles[domain_num].XLAT.values)
    max_lon = np.max(handles[domain_num].XLONG.values)
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


def impute_domain(handles, domain_num, overlap=50):
    """Impute smaller domain in largest domain

    Parameters
    ----------
    handles : list
        List of xarray.Dataset objects for each domain
    domain_num : int
        Domain number to stitch into largest domain
    overlap : int
        Number of grid points to use for blending edges

    Returns
    -------
    handles : list
        List of xarray.Dataset objects for each domain
    """
    out = get_domain_region(handles, domain_num)
    (lat_range, lon_range, min_lat, max_lat, min_lon,
     max_lon, n_lats, n_lons) = out
    regridder = Regridder(handles[domain_num].XLAT[0],
                          handles[domain_num].XLONG[0],
                          min_lat, max_lat, min_lon, max_lon, n_lats, n_lons)
    handles[domain_num] = regridder.regrid_data(handles[domain_num])
    for field in handles[0]:
        if field not in ['Times']:
            arr1 = handles[0][field].loc[dict(south_north=lat_range,
                                              west_east=lon_range)]
            arr2 = handles[domain_num][field]
            out = blend_domains(arr1, arr2, overlap=overlap)
            handles[0][field].loc[dict(south_north=lat_range,
                                       west_east=lon_range)] = out
    return handles


def stitch_domains(year, month, time_step, input_files, overlap=50,
                   n_domains=4, max_level=15):
    """Stitch all smaller domains into largest domain

    Parameters
    ----------
    year : int
        Year for input files
    month : int
        Month for input files
    time_step : int
        Time step for input files for the specified month. e.g. if year=2017,
        month=3, time_step=0 this will select the file for the first time step
        of 2017-03-01. If None then stitch and save will be done for full
        month.
    input_files : dict
        Dictionary of input files with keys corresponding to domain number
    overlap : int
        Number of grid points to use for blending edges
    n_domains : int
        Number of domains to stitch together
    max_level : int
        Max pressure level index

    Returns
    -------
    handles : list
        List of xarray.Dataset objects with smaller domains stitched into
        handles[0]
    """
    logger.info(f'Getting domain files for year={year}, month={month},'
                f' timestep={time_step}.')
    step_files = [input_files[d][time_step] for d in range(n_domains)]
    logger.info(f'Getting data handles for files: {step_files}')
    handles = get_handles(step_files)
    logger.info('Unstaggering variables for all handles')
    handles = unstagger_vars(handles)
    logger.info(f'Pruning pressure levels to level={max_level}')
    handles = prune_levels(handles, max_level=max_level)
    logger.info(f'Regridding main domain for year={year}, month={month}, '
                f'timestep={time_step}')
    handles = regrid_main_domain(handles)
    for j in range(1, n_domains):
        logger.info(f'Imputing domain {j + 1} for year={year}, '
                    f'month={month}, timestep={time_step}')
        handles = impute_domain(handles, j, overlap=overlap)
    return handles


def stitch_and_save(year, month, input_pattern, out_pattern,
                    time_step=None, overlap=50, n_domains=4, max_level=15,
                    overwrite=False):
    """Stitch all smaller domains into largest domain and save output

    Parameters
    ----------
    year : int
        Year for input files
    month : int
        Month for input files
    time_step : int
        Time step for input files for the specified month. e.g. if year=2017,
        month=3, time_step=0 this will select the file for the first time step
        of 2017-03-01. If None then stitch and save will be done for full
        month.
    input_pattern : str
        Pattern for input files. Assumes pattern contains {month}, {year}, and
        {domain}
    out_pattern : str
        Pattern for output files
    overlap : int
        Number of grid points to use for blending edges
    n_domains : int
        Number of domains to stitch together
    max_level : int
        Max pressure level index
    overwrite : bool
        Whether to overwrite existing files
    """
    logger.info(f'Getting file patterns for year={year}, month={month}')
    input_files, out_files = get_files(year, month, input_pattern,
                                       out_pattern, n_domains=n_domains)
    out_files = (out_files if time_step is None
                 else out_files[time_step - 1: time_step])
    for i, out_file in enumerate(out_files):
        if not os.path.exists(out_file) or overwrite:
            handles = stitch_domains(year, month, i, input_files,
                                     overlap=overlap, n_domains=n_domains,
                                     max_level=max_level)
            basedir = os.path.dirname(out_file)
            os.makedirs(basedir, exist_ok=True)
            handles[0].to_netcdf(out_file)
            logger.info(f'Saved stitched file to {out_file}')
