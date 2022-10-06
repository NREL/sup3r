# -*- coding: utf-8 -*-
"""Utilities used for QA"""
import numpy as np
from PIL import Image


def tke_frequency_spectrum(u, v):
    """Turbulent Kinetic Energy Spectrum. Gives the portion of kinetic energy
    associated with each wavenumber.

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
        with a given frequency
    """
    v_f = np.fft.fftn(np.mean(v, axis=(0, 1)))
    u_f = np.fft.fftn(np.mean(u, axis=(0, 1)))
    E_f = np.abs(v_f)**2 + np.abs(u_f)**2
    n_steps = E_f.shape[0] // 2
    E_f_a = E_f[:n_steps]
    E_f_b = E_f[-n_steps:][::-1]
    E_f = E_f_a + E_f_b
    return E_f


def tke_wavenumber_spectrum(u, v, k_range=None, axis=0):
    """Turbulent Kinetic Energy Spectrum. Gives the portion of kinetic energy
    associated with each wavenumber.

    Parameters
    ----------
    u: ndarray
        (lat, lon)
        U component of wind
    v : ndarray
        (lat, lon)
        V component of wind
    k_range : list | None
        List with min and max wavenumber. When comparing spectra for different
        domains this needs to be tailored to the specific domain.  e.g. k =
        [1/max_length, ..., 1/min_length] If this is not specified k with be
        set to [0, ..., len(y)] where y is the fft output.
    axis : int
        Axis to average over to get a 1D wind field. If axis=0 this returns
        the zonal energy spectrum

    Returns
    -------
    ndarray
        1D array of amplitudes corresponding to the portion of total energy
        with a given wavenumber
    """
    u_k = np.fft.fftn(u)
    v_k = np.fft.fftn(v)
    E_k = np.mean(np.abs(v_k)**2 + np.abs(u_k)**2, axis=axis)
    if k_range is None:
        k = np.arange(len(E_k))
    else:
        k = np.linspace(k_range[0], k_range[1], len(E_k))
    n_steps = len(k) // 2
    E_k = k**2 * E_k
    E_k_a = E_k[1:n_steps + 1]
    E_k_b = E_k[-n_steps:][::-1]
    E_k = E_k_a + E_k_b
    return k[:n_steps], E_k


def tke_series(u, v):
    """Longitudinal Turbulent Kinetic Energy Spectrum time series. Gives the
    mean tke spectrum over time.

    Parameters
    ----------
    u: ndarray
        (lat, lon, time)
        U component of wind
    v : ndarray
        (lat, lon, time)
        V component of wind

    Returns
    -------
    ndarray
        1D array of mean tke amplitudes over time
    """

    return [np.mean(tke_wavenumber_spectrum(u[..., t], v[..., t]))
            for t in range(u.shape[-1])]


def velocity_gradient_dist(u, bins=50, range=None, diff_max=7, scale=1):
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
    diff_max : float
        Max value to keep for velocity gradient

    Returns
    -------
    ndarray
        Normalized du / dx at bin centers
    ndarray
        Normalized du / dx value counts
    float
        Normalization factor
    """
    diffs = np.diff(u, axis=1).flatten()
    diffs /= scale
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.sum()
    return centers, counts, norm


def vorticity_dist(u, v, bins=50, range=None, diff_max=14, scale=1):
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
    diff_max : float
        Max value to keep for vorticity

    Returns
    -------
    ndarray
        Normalized vorticity values at bin centers
    ndarray
        Normalized vorticity value counts
    float
        Normalization factor
    """
    dudy = np.diff(u, axis=0, append=np.mean(u)).flatten()
    dvdx = np.diff(v, axis=1, append=np.mean(v)).flatten()
    diffs = dudy - dvdx
    diffs /= scale
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.sum()
    return centers, counts, norm


def ws_ramp_rate_dist(u, v, bins=50, range=None, diff_max=10, t_steps=1,
                      scale=1):
    """Returns the windspeed ramp rate distribution.

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
    diff_max : float
        Max value to keep for ramp rate
    t_steps : int
        Number of time steps to use for differences. e.g. If t_steps=1 this
        uses ws[i + 1] - [i] to compute ramp rates.

    Returns
    -------
    ndarray
        d(ws) / dt values at bin centers
    ndarray
        Normalized d(ws) / dt value counts
    float
        Normalization factor
    """
    msg = (f'Received t_steps={t_steps} for ramp rate calculation but data '
           f'only has {u.shape[-1]} time steps')
    assert t_steps < u.shape[-1], msg
    ws = np.hypot(u, v)
    diffs = (ws[..., t_steps:] - ws[..., :-t_steps]).flatten()
    diffs /= scale
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.sum()
    return centers, counts, norm


def spatial_interp(low, s_enhance):
    """Spatial bilinear interpolation for low resolution field. Used to provide
    baseline for comparison with gan output

    Parameters
    ----------
    low : ndarray
        Low resolution field to interpolate.
        (spatial_1, spatial_2, temporal)
    s_enhance : int
        Factor by which to enhance the spatial domain

    Returns
    -------
    ndarray
        Spatially interpolated low resolution output
    """

    high = np.zeros((low.shape[0] * s_enhance, low.shape[1] * s_enhance,
                     low.shape[-1]))
    for t in range(low.shape[-1]):
        im = Image.fromarray(low[..., t])
        im = im.resize((low[..., t].shape[1] * s_enhance,
                        low[..., t].shape[0] * s_enhance),
                       resample=Image.Resampling.BILINEAR)
        high[..., t] = np.array(im)
    return high


def temporal_interp(low, t_enhance):
    """Temporal bilinear interpolation for low resolution field. Used to
    provide baseline for comparison with gan output

    Parameters
    ----------
    low : ndarray
        Low resolution field to interpolate.
        (spatial_1, spatial_2, temporal)
    t_enhance : int
        Factor by which to enhance the temporal domain

    Returns
    -------
    ndarray
        Temporally interpolated low resolution output
    """

    high = np.zeros((low.shape[0], low.shape[1], low.shape[-1] * t_enhance))
    for t in range(high.shape[-1]):
        t0 = t // t_enhance
        t1 = t0 + 1
        alpha = (t / t_enhance - t0) / (t1 - t0)
        if t1 == low.shape[-1]:
            tmp = low[..., t0]
        else:
            tmp = (1 - alpha) * low[..., t0] + alpha * low[..., t1]
        high[..., t] = tmp
    return high


def st_interp(low, s_enhance, t_enhance):
    """Spatiotemporal bilinear interpolation for low resolution field. Used to
    provide baseline for comparison with gan output

    Parameters
    ----------
    low : ndarray
        Low resolution field to interpolate.
        (spatial_1, spatial_2, temporal)
    s_enhance : int
        Factor by which to enhance the spatial domain
    t_enhance : int
        Factor by which to enhance the temporal domain

    Returns
    -------
    ndarray
        Spatiotemporally interpolated low resolution output
    """
    return temporal_interp(spatial_interp(low, s_enhance), t_enhance)
