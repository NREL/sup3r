"""Utilities used across multiple test files"""
import os
import numpy as np
import xarray as xr

from sup3r.postprocessing.file_handling import OutputHandlerH5
from sup3r.utilities.utilities import pd_date_range


def make_fake_nc_files(td, input_file, n_files):
    """Make dummy nc files with increasing times

    Parameters
    ----------
    input_file : str
        File to use as template for all dummy files
    n_files : int
        Number of dummy files to create

    Returns
    -------
    fake_files : list
        List of dummy files
    """
    fake_dates = [f'2014-10-01_{str(i).zfill(2)}_00_00'
                  for i in range(n_files)]
    fake_times = [f'2014-10-01 {str(i).zfill(2)}:00:00'
                  for i in range(n_files)]
    fake_files = [os.path.join(td, f'input_{date}') for date in fake_dates]
    for i in range(n_files):
        input_dset = xr.open_dataset(input_file)
        with xr.Dataset(input_dset) as dset:
            dset['Times'][:] = np.array([fake_times[i].encode('ASCII')],
                                        dtype='|S19')
            dset['XTIME'][:] = i
            dset.to_netcdf(fake_files[i])
    return fake_files


def make_fake_h5_chunks(td):
    """Make fake h5 chunked output files for a 5x spatial 2x temporal
    multi-node forward pass output.

    Parameters
    ----------
    td : tempfile.TemporaryDirectory
        Test TemporaryDirectory

    Returns
    -------
    out_files : list
        List of filepaths to chunked files.
    data : ndarray
        (spatial_1, spatial_2, temporal, features)
        High resolution forward pass output
    ws_true : ndarray
        Windspeed between 0 and 20 in shape (spatial_1, spatial_2, temporal, 1)
    wd_true : ndarray
        Windir between 0 and 360 in shape (spatial_1, spatial_2, temporal, 1)
    features : list
        List of feature names corresponding to the last dimension of data
        ['windspeed_100m', 'winddirection_100m']
    slices_lr : list
        List of low res temporal slices corresponding to the out_files list
    slices_hr : list
        List of high res temporal slices corresponding to the out_files list
    low_res_lat_lon : ndarray
        Array of lat/lon for input data. (spatial_1, spatial_2, 2)
        Last dimension has ordering (lat, lon)
    low_res_times : list
        List of np.datetime64 objects for coarse data.
    """

    features = ['windspeed_100m', 'winddirection_100m']
    model_meta_data = {'foo': 'bar'}
    shape = (50, 50, 96, 1)
    ws_true = np.random.uniform(0, 20, shape)
    wd_true = np.random.uniform(0, 360, shape)
    data = np.concatenate((ws_true, wd_true), axis=3)
    lat = np.linspace(90, 0, 10)
    lon = np.linspace(-180, 0, 10)
    lon, lat = np.meshgrid(lon, lat)
    low_res_lat_lon = np.dstack((lat, lon))

    low_res_times = pd_date_range('20220101', '20220103', freq='3600s',
                                  inclusive='left')

    slices_lr = [slice(0, 24), slice(24, None)]
    slices_hr = [slice(0, 48), slice(48, None)]

    out_files = [os.path.join(td, 'fp_out_0.h5'),
                 os.path.join(td, 'fp_out_1.h5')]

    for i, (slice_lr, slice_hr) in enumerate(zip(slices_lr, slices_hr)):
        OutputHandlerH5.write_output(data[:, :, slice_hr, :], features,
                                     low_res_lat_lon,
                                     low_res_times[slice_lr],
                                     out_files[i],
                                     model_meta_data,
                                     max_workers=1)

    out = (out_files, data, ws_true, wd_true, features, slices_lr, slices_hr,
           low_res_lat_lon, low_res_times)

    return out


def tke_spectrum(u, v):
    """Longitudinal Turbulent Kinetic Energy Spectrum. Gives the portion of
    kinetic energy associated with each longitudinal wavenumber.

    Parameters
    ----------
    u: ndarray
        (lat, lon)
        U component of wind
    v : ndarray
        (lat, lon)
        V component of wind

    Returns
    -------
    ndarray
        1D array of amplitudes corresponding to the portion of total energy
        with a given longitudinal wavenumber
    """
    v_k = np.fft.fftn(v)
    u_k = np.fft.fftn(u)
    E_k = np.abs(v_k)**2 + np.abs(u_k)**2
    E_k = np.mean(E_k, axis=0)
    E_k *= np.arange(E_k.shape[0])**2
    n_steps = E_k.shape[0] // 2
    E_k_a = E_k[:n_steps]
    E_k_b = E_k[-n_steps:][::-1]
    E_k = E_k_a + E_k_b
    return E_k


def velocity_gradient_dist(u, bins=50, range=None):
    """Returns the longitudinal velocity gradient distribution.

    Parameters
    ----------
    u: ndarray
        Longitudinal velocity component
        (lat, lon, temporal)
    bins : int
        Number of bins for the velocity gradient pdf.
    range : tuple | None
        Optional min/max range for the velocity gradient pdf.

    Returns
    -------
    ndarray
        du/dx at bin centers
    ndarray
        Normalized delta_u / delta_x value counts
    """
    diffs = np.diff(u, axis=1).flatten()
    diffs = diffs[(np.abs(diffs) < 7)]
    diffs = diffs / np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.max()
    return centers, counts


def vorticity_dist(u, v, bins=50, range=None):
    """Returns the vorticity distribution.

    Parameters
    ----------
    u: ndarray
        Longitudinal velocity component
        (lat, lon, temporal)
    v : ndarray
        Latitudinal velocity component
        (lat, lon, temporal)
    bins : int
        Number of bins for the vorticity pdf.
    range : tuple | None
        Optional min/max range for the vorticity pdf.

    Returns
    -------
    ndarray
        vorticity values at bin centers
    ndarray
        Normalized vorticity value counts
    """
    dudy = np.diff(u, axis=0, append=np.mean(u)).flatten()
    dvdx = np.diff(v, axis=1, append=np.mean(v)).flatten()
    diffs = dudy - dvdx
    diffs = diffs[(np.abs(diffs) < 14)]
    diffs = diffs / np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.max()
    return centers, counts


def ramp_rate_dist(u, v, bins=50, range=None):
    """Returns the ramp rate distribution.

    Parameters
    ----------
    u: ndarray
        Longitudinal velocity component
        (lat, lon, temporal)
    v : ndarray
        Latitudinal velocity component
        (lat, lon, temporal)
    bins : int
        Number of bins for the ramp rate pdf.
    range : tuple | None
        Optional min/max range for the ramp rate pdf.

    Returns
    -------
    ndarray
        dws/dt values at bin centers
    ndarray
        Normalized delta_ws / delta_t value counts
    """
    diffs = np.diff(np.sqrt(u**2 + v**2), axis=-1).flatten()
    diffs = diffs[(np.abs(diffs) < 10)]
    diffs = diffs / np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.max()
    return centers, counts
