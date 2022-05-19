"""
Loss metrics for Sup3r
"""

import tensorflow as tf


def gaussian_kernel(x1, x2, beta=1.0):
    """Gaussian kernel for mmd content loss

    Parameters
    ----------
    x1: tf.tensor
        synthetic generator output
    x2: tf.tensor
        high resolution data

    Returns
    -------
    tf.tensor
        kernel output tensor
    """
    return tf.exp(-beta * (x1 - x2)**2)


def max_mean_discrepancy(x1, x2, beta=1.0):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.

    Parameters
    ----------
    x1: tf.tensor
        synthetic generator output
        (n_observations, spatial_1, spatial_2, temporal, features)
    x2: tf.tensor
        high resolution data
        (n_observations, spatial_1, spatial_2, temporal, features)
    beta : float
        scaling parameter for gaussian kernel

    Returns
    tf.tensor
        tensor with content loss value summed over feature channel
        (n_observations, spatial_1, spatial_2, temporal)
    -------


    """
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = tf.reduce_sum(x1x1 + x2x2 - 2 * x1x2, axis=-1)
    return diff
