"""Loss metrics for Sup3r"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError


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
    sigma : float
        Standard deviation for gaussian kernel

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
        mmd = self.MMD_LOSS(x1, x2, sigma=sigma)
        mse = self.MSE_LOSS(x1, x2)
        return (mmd + mse) / 2


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


class SpatialExtremesOnlyLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the min/max values in the
    spatial domain. This does not include an additional MAE term"""

    MAE_LOSS = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Custom content loss that encourages temporal min/max accuracy

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x1_min = tf.reduce_min(x1, axis=(1, 2))
        x2_min = tf.reduce_min(x2, axis=(1, 2))

        x1_max = tf.reduce_max(x1, axis=(1, 2))
        x2_max = tf.reduce_max(x2, axis=(1, 2))

        mae_min = self.MAE_LOSS(x1_min, x2_min)
        mae_max = self.MAE_LOSS(x1_max, x2_max)

        return (mae_min + mae_max) / 2


class SpatialExtremesLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the min/max values in the
    spatial domain"""

    MAE_LOSS = MeanAbsoluteError()
    EX_LOSS = SpatialExtremesOnlyLoss()

    def __init__(self, weight=1.0):
        """Initialize the loss with given weight

        Parameters
        ----------
        weight : float
            Weight for min/max loss terms. Setting this to zero turns
            loss into MAE.
        """
        super().__init__()
        self._weight = weight

    def __call__(self, x1, x2):
        """Custom content loss that encourages temporal min/max accuracy

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        mae = self.MAE_LOSS(x1, x2)
        ex_mae = self.EX_LOSS(x1, x2)

        return (mae + 2 * self._weight * ex_mae) / 3


class TemporalExtremesOnlyLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the min/max values in the
    timeseries. This does not include an additional mae term"""

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

        mae_min = self.MAE_LOSS(x1_min, x2_min)
        mae_max = self.MAE_LOSS(x1_max, x2_max)

        return (mae_min + mae_max) / 2


class TemporalExtremesLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the min/max values in the
    timeseries"""

    MAE_LOSS = MeanAbsoluteError()
    EX_LOSS = TemporalExtremesOnlyLoss()

    def __init__(self, weight=1.0):
        """Initialize the loss with given weight

        Parameters
        ----------
        weight : float
            Weight for min/max loss terms. Setting this to zero turns
            loss into MAE.
        """
        super().__init__()
        self._weight = weight

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
        mae = self.MAE_LOSS(x1, x2)
        ex_mae = self.EX_LOSS(x1, x2)

        return (mae + 2 * self._weight * ex_mae) / 3


class SpatiotemporalExtremesLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the min/max values across both
    space and time"""

    MAE_LOSS = MeanAbsoluteError()
    S_EX_LOSS = SpatialExtremesOnlyLoss()
    T_EX_LOSS = TemporalExtremesOnlyLoss()

    def __init__(self, spatial_weight=1.0, temporal_weight=1.0):
        """Initialize the loss with given weight

        Parameters
        ----------
        spatial_weight : float
            Weight for spatial min/max loss terms.
        temporal_weight : float
            Weight for temporal min/max loss terms.
        """
        super().__init__()
        self.s_weight = spatial_weight
        self.t_weight = temporal_weight

    def __call__(self, x1, x2):
        """Custom content loss that encourages spatiotemporal min/max accuracy.

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
        mae = self.MAE_LOSS(x1, x2)
        s_ex_mae = self.S_EX_LOSS(x1, x2)
        t_ex_mae = self.T_EX_LOSS(x1, x2)
        return (mae + 2 * self.s_weight * s_ex_mae
                + 2 * self.t_weight * t_ex_mae) / 5


class SpatialFftOnlyLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the spatial frequency spectrum"""

    MAE_LOSS = MeanAbsoluteError()

    @staticmethod
    def _freq_weights(x):
        """Get product of squared frequencies to weight frequency amplitudes"""
        k0 = np.array([k**2 for k in range(x.shape[1])])
        k1 = np.array([k**2 for k in range(x.shape[2])])
        freqs = np.multiply.outer(k0, k1)
        freqs = tf.convert_to_tensor(freqs[np.newaxis, ..., np.newaxis])
        return tf.cast(freqs, x.dtype)

    def _fft(self, x):
        """Apply needed transpositions and fft operation."""
        x_hat = tf.transpose(x, perm=[3, 0, 1, 2])
        x_hat = tf.signal.fft2d(tf.cast(x_hat, tf.complex64))
        x_hat = tf.transpose(x_hat, perm=[1, 2, 3, 0])
        x_hat = tf.cast(tf.abs(x_hat), x.dtype)
        x_hat = tf.math.multiply(self._freq_weights(x), x_hat)
        return tf.math.log(1 + x_hat)

    def __call__(self, x1, x2):
        """Custom content loss that encourages frequency domain accuracy

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x1_hat = self._fft(x1)
        x2_hat = self._fft(x2)
        return self.MAE_LOSS(x1_hat, x2_hat)


class SpatiotemporalFftOnlyLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the spatiotemporal frequency
    spectrum"""

    MAE_LOSS = MeanAbsoluteError()

    @staticmethod
    def _freq_weights(x):
        """Get product of squared frequencies to weight frequency amplitudes"""
        k0 = np.array([k**2 for k in range(x.shape[1])])
        k1 = np.array([k**2 for k in range(x.shape[2])])
        f = np.array([f**2 for f in range(x.shape[3])])
        freqs = np.multiply.outer(k0, k1)
        freqs = np.multiply.outer(freqs, f)
        freqs = tf.convert_to_tensor(freqs[np.newaxis, ..., np.newaxis])
        return tf.cast(freqs, x.dtype)

    def _fft(self, x):
        """Apply needed transpositions and fft operation."""
        x_hat = tf.transpose(x, perm=[4, 0, 1, 2, 3])
        x_hat = tf.signal.fft3d(tf.cast(x_hat, tf.complex64))
        x_hat = tf.transpose(x_hat, perm=[1, 2, 3, 4, 0])
        x_hat = tf.cast(tf.abs(x_hat), x.dtype)
        x_hat = tf.math.multiply(self._freq_weights(x), x_hat)
        return tf.math.log(1 + x_hat)

    def __call__(self, x1, x2):
        """Custom content loss that encourages frequency domain accuracy

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
        x1_hat = self._fft(x1)
        x2_hat = self._fft(x2)
        return self.MAE_LOSS(x1_hat, x2_hat)


class StExtremesFftLoss(tf.keras.losses.Loss):
    """Loss class that encourages accuracy of the min/max values across both
    space and time as well as frequency domain accuracy."""

    def __init__(self, spatial_weight=1.0, temporal_weight=1.0,
                 fft_weight=1.0):
        """Initialize the loss with given weight

        Parameters
        ----------
        spatial_weight : float
            Weight for spatial min/max loss terms.
        temporal_weight : float
            Weight for temporal min/max loss terms.
        fft_weight : float
            Weight for the fft loss term.
        """
        super().__init__()
        self.st_ex_loss = SpatiotemporalExtremesLoss(spatial_weight,
                                                     temporal_weight)
        self.fft_loss = SpatiotemporalFftOnlyLoss()
        self.fft_weight = fft_weight

    def __call__(self, x1, x2):
        """Custom content loss that encourages spatiotemporal min/max accuracy
        and fft accuracy.

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
        return (5 * self.st_ex_loss(x1, x2)
                + self.fft_weight * self.fft_loss(x1, x2)) / 6
