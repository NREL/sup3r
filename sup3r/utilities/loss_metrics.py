"""
Loss metrics for Sup3r
"""

import tensorflow as tf


def max_mean_discrepancy(x, y, kernel='rbf'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx = tf.matmul(x, tf.transpose(x))
    yy = tf.matmul(y, tf.transpose(y))
    xy = tf.matmul(x, tf.transpose(y))
    rx = tf.broadcast_to(tf.expand_dims(tf.linalg.diag_part(xx), 0), xx.shape)
    ry = tf.broadcast_to(tf.expand_dims(tf.linalg.diag_part(yy), 0), yy.shape)

    dxx = tf.transpose(rx) + rx - 2. * xx
    dyy = tf.transpose(ry) + ry - 2. * yy
    dxy = tf.transpose(rx) + ry - 2. * xy

    XX, YY, XY = (tf.zeros(xx.shape), tf.zeros(xx.shape), tf.zeros(xx.shape))

    if kernel == 'linear':
        XX = xx
        YY = yy
        XY = xy

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [1, 10, 15, 20, 50]
        for a in bandwidth_range:
            XX += tf.exp(-0.5 * dxx / a)
            YY += tf.exp(-0.5 * dyy / a)
            XY += tf.exp(-0.5 * dxy / a)

    return tf.reduce_mean(XX + YY - 2. * XY, axis=0)
