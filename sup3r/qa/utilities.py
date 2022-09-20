"""Utilities used for QA"""
import numpy as np
from PIL import Image


def tke_spectrum(u, v, axis=0):
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
    axis : int
        Axis to average over to get a 1D wind field. If axis=0 this returns
        the zonal energy spectrum

    Returns
    -------
    ndarray
        1D array of amplitudes corresponding to the portion of total energy
        with a given wavenumber
    """
    v_k = np.fft.fftn(np.mean(v, axis=axis))
    u_k = np.fft.fftn(np.mean(u, axis=axis))
    E_k = np.abs(v_k)**2 + np.abs(u_k)**2
    E_k *= np.arange(E_k.shape[0])**2
    n_steps = E_k.shape[0] // 2
    E_k_a = E_k[:n_steps]
    E_k_b = E_k[-n_steps:][::-1]
    E_k = E_k_a + E_k_b
    return E_k


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

    return [np.mean(tke_spectrum(u[..., t], v[..., t]))
            for t in range(u.shape[-1])]


def velocity_gradient_dist(u, bins=50, range=None, diff_max=7):
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
        du/dx at bin centers
    ndarray
        Normalized delta_u / delta_x value counts
    """
    diffs = np.diff(u, axis=1).flatten()
    diffs = diffs[(np.abs(diffs) < diff_max)]
    diffs = diffs / np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.sum()
    return centers, counts


def vorticity_dist(u, v, bins=50, range=None, diff_max=14):
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
        vorticity values at bin centers
    ndarray
        Normalized vorticity value counts
    """
    dudy = np.diff(u, axis=0, append=np.mean(u)).flatten()
    dvdx = np.diff(v, axis=1, append=np.mean(v)).flatten()
    diffs = dudy - dvdx
    diffs = diffs[(np.abs(diffs) < diff_max)]
    diffs = diffs / np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.sum()
    return centers, counts


def ramp_rate_dist(u, v, bins=50, range=None, diff_max=10):
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
    diff_max : float
        Max value to keep for ramp rate

    Returns
    -------
    ndarray
        dws/dt values at bin centers
    ndarray
        Normalized delta_ws / delta_t value counts
    """
    diffs = np.diff(np.sqrt(u**2 + v**2), axis=-1).flatten()
    diffs = diffs[(np.abs(diffs) < diff_max)]
    diffs = diffs / np.sqrt(np.mean(diffs**2))
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    counts = counts.astype(float) / counts.sum()
    return centers, counts


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
