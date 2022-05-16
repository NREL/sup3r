"""
Loss metrics for Sup3r
"""

import tensorflow as tf


def gaussian_kernel(x1, x2, beta=1.0):
    """Gaussian kernel for mmd content loss

    Parameters
    ----------
    x1: tf.tensor
        first sample, distribution P
    x2: tf.tensor
        second sample, distribution Q

    Returns
    -------
    tf.tensor
    """
    return tf.reduce_sum(tf.exp(-beta * (x1 - x2)**2), axis=-1)


def max_mean_discrepancy(x1, x2, beta=1.0):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.

    Parameters
    ----------
    x1: tf.tensor
        first sample, distribution P
    x2: tf.tensor
        second sample, distribution Q
    beta : float
        scaling parameter for gaussian kernel

    Returns
    tf.tensor
        0D tensor with content loss value
    -------


    """
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = tf.reduce_mean(x1x1)
    diff -= 2 * tf.reduce_mean(x1x2)
    diff += tf.reduce_mean(x2x2)
    return diff
