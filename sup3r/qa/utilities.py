# -*- coding: utf-8 -*-
"""Utilities used for QA"""
import numpy as np
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


def tke_frequency_spectrum(u, v):
    """Kinetic Energy Spectrum. Gives the portion of kinetic energy
    associated with each frequency.

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


def frequency_spectrum(var):
    """Frequency Spectrum. Gives the portion of the variable
    associated with each frequency.

    Parameters
    ----------
    var: ndarray
        (lat, lon, temporal)

    Returns
    -------
    ndarray
        1D array of amplitudes corresponding to the portion of the variable
        with a given frequency
    """
    var_f = np.fft.fftn(np.mean(var, axis=(0, 1)))
    E_f = np.abs(var_f)**2
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


def wavenumber_spectrum(var, k_range=None, axis=0):
    """Wavenumber Spectrum. Gives the portion of the given variable
    associated with each wavenumber.

    Parameters
    ----------
    var: ndarray
        (lat, lon)
    k_range : list | None
        List with min and max wavenumber. When comparing spectra for different
        domains this needs to be tailored to the specific domain.  e.g. k =
        [1/max_length, ..., 1/min_length] If this is not specified k with be
        set to [0, ..., len(y)] where y is the fft output.
    axis : int
        Axis to average over to get a 1D field. If axis=0 this returns
        the zonal spectrum

    Returns
    -------
    ndarray
        1D array of amplitudes corresponding to the portion of the given
        variable with a given wavenumber
    """
    var_k = np.fft.fftn(var)
    E_k = np.mean(np.abs(var_k)**2, axis=axis)
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


def gradient_dist(var, bins=40, range=None, diff_max=None, scale=1):
    """Returns the gradient distribution for the given variable.

    Parameters
    ----------
    var: ndarray
        (lat, lon, temporal)
    bins : int
        Number of bins for the gradient pdf.
    range : tuple | None
        Optional min/max range for the gradient pdf.
    diff_max : float
        Max value to keep for gradient

    Returns
    -------
    ndarray
        d(var) / dx at bin centers
    ndarray
        Normalized d(var) / dx value counts
    float
        Normalization factor
    """
    diffs = np.diff(var, axis=1).flatten()
    diffs /= scale
    diff_max = diff_max or np.percentile(np.abs(diffs), 95)
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, centers = continuous_dist(diffs, bins=bins, range=range)
    return centers, counts, norm


def vorticity_dist(u, v, bins=40, range=None, diff_max=None, scale=1):
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
    float
        Normalization factor
    """
    dudy = np.diff(u, axis=0, append=np.mean(u)).flatten()
    dvdx = np.diff(v, axis=1, append=np.mean(v)).flatten()
    diffs = dudy - dvdx
    diffs /= scale
    diff_max = diff_max or np.percentile(np.abs(diffs), 95)
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, centers = continuous_dist(diffs, bins=bins, range=range)
    return centers, counts, norm


def ramp_rate_dist(var, bins=40, range=None, diff_max=None, t_steps=1,
                   scale=1):
    """Returns the ramp rate distribution for the given variable.

    Parameters
    ----------
    var: ndarray
        (lat, lon, temporal)
    bins : int
        Number of bins for the ramp rate pdf.
    range : tuple | None
        Optional min/max range for the ramp rate pdf.
    diff_max : float
        Max value to keep for ramp rate
    t_steps : int
        Number of time steps to use for differences. e.g. If t_steps=1 this
        uses var[i + 1] - [i] to compute ramp rates.

    Returns
    -------
    ndarray
        d(var) / dt values at bin centers
    ndarray
        Normalized d(var) / dt value counts
    float
        Normalization factor
    """

    msg = (f'Received t_steps={t_steps} for ramp rate calculation but data '
           f'only has {var.shape[-1]} time steps')
    assert t_steps < var.shape[-1], msg
    diffs = (var[..., t_steps:] - var[..., :-t_steps]).flatten()
    diffs /= scale
    diff_max = diff_max or np.percentile(np.abs(diffs), 95)
    diffs = diffs[(np.abs(diffs) < diff_max)]
    norm = np.sqrt(np.mean(diffs**2))
    counts, centers = continuous_dist(diffs, bins=bins, range=range)
    return centers, counts, norm


def continuous_dist(diffs, bins=None, range=None):
    """Get interpolated distribution from histogram

    Parameters
    ----------
    diffs : ndarray
        Array of values to use to construct distribution
    bins : int
        Number of bins for the distribution. If None then the number of bins
        will be determined from the value range and the smallest difference
        between values
    range : tuple | None
        Optional min/max range for the distribution.

    Returns
    -------
    ndarray
        distribution value counts
    ndarray
        distribution values at bin centers
    """
    if bins is None:
        dx = np.abs(np.diff(diffs))
        dx = dx[dx > 0]
        dx = np.mean(dx)
        bins = int((np.max(diffs) - np.min(diffs)) / dx)
        logger.debug(f'Using n_bins={bins} to compute distribution')
    counts, edges = np.histogram(diffs, bins=bins, range=range)
    centers = edges[:-1] + (np.diff(edges) / 2)
    indices = np.where(counts > 0)
    y = counts[indices]
    x = centers[indices]
    interp = interp1d(x, y, bounds_error=False, fill_value=None)
    counts = interp(centers)
    counts = counts.astype(float) / counts.sum()
    return counts, centers
