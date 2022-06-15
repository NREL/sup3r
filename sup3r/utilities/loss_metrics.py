"""
Loss metrics for Sup3r
"""

from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf


def gaussian_kernel(x1, x2, sigma=1.0):
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

    References
    ----------
    Following MMD implementation in https://github.com/lmjohns3/theanets
    """

    result = tf.exp(-0.5 * tf.reduce_sum(
        (tf.expand_dims(x1, axis=1) - x2)**2, axis=-1) / sigma**2)
    return result


class ExpLoss(tf.keras.losses.Loss):
    def __call__(self, x1, x2):
        """
        Exponential difference loss function

        Parameters
        ----------
        x1: tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2: tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        tf.tensor
            tensor with content loss value summed over feature channel
            (n_observations, spatial_1, spatial_2, temporal)
        """
        diff = tf.reduce_sum(1 - tf.exp(-(x1 - x2)**2), axis=-1)
        return diff


class MmdMseLoss(tf.keras.losses.Loss):

    MSE_LOSS = MeanSquaredError()

    def __call__(self, x1, x2, sigma=1.0):
        """Maximum mean discrepancy (MMD) based on Gaussian kernel function
        for keras models

        Parameters
        ----------
        x1: tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2: tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)
        sigma : float
            standard deviation for gaussian kernel

        Returns
        tf.tensor
            tensor with content loss value summed over feature channel
            (n_observations, spatial_1, spatial_2, temporal)
        """
        x1x1 = gaussian_kernel(x1, x1, sigma)
        x2x2 = gaussian_kernel(x2, x2, sigma)
        x1x2 = gaussian_kernel(x1, x2, sigma)
        mmd = tf.reduce_mean(x1x1 + x2x2 - 2 * x1x2)
        mse = self.MSE_LOSS(x1, x2)
        return mmd + mse
