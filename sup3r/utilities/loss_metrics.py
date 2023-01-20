"""Loss metrics for Sup3r"""

from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import tensorflow as tf


def gaussian_kernel(x1, x2, sigma=1.0):
    """Gaussian kernel for mmd content loss

    Parameters
    ----------
    x1 : tf.tensor
        synthetic generator output
        (n_obs, spatial_1, spatial_2, temporal, features)
    x2 : tf.tensor
        high resolution data
        (n_obs, spatial_1, spatial_2, temporal, features)

    Returns
    -------
    tf.tensor
        kernel output tensor

    References
    ----------
    Following MMD implementation in https://github.com/lmjohns3/theanets
    """

    # The expand dims + subtraction compares every entry for the dimension
    # prior to the expanded dimension to every other entry. So expand_dims with
    # axis=1 will compare every observation along axis=0 to every other
    # observation along axis=0.
    result = tf.exp(-0.5 * tf.reduce_sum(
        (tf.expand_dims(x1, axis=1) - x2)**2, axis=-1) / sigma**2)
    return result


class ExpLoss(tf.keras.losses.Loss):
    """Loss class for squared exponential difference"""

    def __call__(self, x1, x2):
        """Exponential difference loss function

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        return tf.reduce_mean(1 - tf.exp(-(x1 - x2)**2))


class MseExpLoss(tf.keras.losses.Loss):
    """Loss class for mse + squared exponential difference"""

    MSE_LOSS = MeanSquaredError()

    def __call__(self, x1, x2):
        """Mse + Exponential difference loss function

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        mse = self.MSE_LOSS(x1, x2)
        exp = tf.reduce_mean(1 - tf.exp(-(x1 - x2)**2))
        return mse + exp


class MmdLoss(tf.keras.losses.Loss):
    """Loss class for max mean discrepancy loss"""

    def __call__(self, x1, x2, sigma=1.0):
        """Maximum mean discrepancy (MMD) based on Gaussian kernel function
        for keras models

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)
        sigma : float
            standard deviation for gaussian kernel

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        mmd = tf.reduce_mean(gaussian_kernel(x1, x1, sigma))
        mmd += tf.reduce_mean(gaussian_kernel(x2, x2, sigma))
        mmd -= tf.reduce_mean(2 * gaussian_kernel(x1, x2, sigma))
        return mmd


class MmdMseLoss(tf.keras.losses.Loss):
    """Loss class for MMD + MSE"""

    MMD_LOSS = MmdLoss()
    MSE_LOSS = MeanSquaredError()

    def __call__(self, x1, x2, sigma=1.0):
        """Maximum mean discrepancy (MMD) based on Gaussian kernel function
        for keras models plus the typical MSE loss.

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)
        sigma : float
            standard deviation for gaussian kernel

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        mmd = self.MMD_LOSS(x1, x2)
        mse = self.MSE_LOSS(x1, x2)
        return mmd + mse


class CoarseMseLoss(tf.keras.losses.Loss):
    """Loss class for coarse mse on spatial average of 5D tensor"""

    MSE_LOSS = MeanSquaredError()

    def __call__(self, x1, x2):
        """Exponential difference loss function

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """

        x1_coarse = tf.reduce_mean(x1, axis=(1, 2))
        x2_coarse = tf.reduce_mean(x2, axis=(1, 2))
        return self.MSE_LOSS(x1_coarse, x2_coarse)


class TemporalExtremesLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the min/max values in the
    timeseries"""

    MAE_LOSS = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Custom content loss that encourages temporal min/max accuracy

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x1_min = tf.reduce_min(x1, axis=3)
        x2_min = tf.reduce_min(x2, axis=3)

        x1_max = tf.reduce_max(x1, axis=3)
        x2_max = tf.reduce_max(x2, axis=3)

        mae = self.MAE_LOSS(x1, x2)
        mae_min = self.MAE_LOSS(x1_min, x2_min)
        mae_max = self.MAE_LOSS(x1_max, x2_max)

        return mae + mae_min + mae_max
