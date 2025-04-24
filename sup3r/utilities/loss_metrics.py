"""Content loss metrics for Sup3r"""

from typing import ClassVar

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError


def _derivative(x, axis=1):
    """Custom derivative function for compatibility with tensorflow.

    Note
    ----
    Matches np.gradient by using the central difference approximation.

    Parameters
    ----------
    x : tf.Tensor
        (n_observations, spatial_1, spatial_2, temporal)
    axis : int
        Axis to take derivative over
    """
    if axis == 1:
        return tf.concat(
            [
                x[:, 1:2] - x[:, 0:1],
                (x[:, 2:] - x[:, :-2]) / 2,
                x[:, -1:] - x[:, -2:-1],
            ],
            axis=axis,
        )
    if axis == 2:
        return tf.concat(
            [
                x[:, :, 1:2] - x[:, :, 0:1],
                (x[:, :, 2:] - x[:, :, :-2]) / 2,
                x[:, :, -1:] - x[:, :, -2:-1],
            ],
            axis=axis,
        )
    if axis == 3:
        return tf.concat(
            [
                x[:, :, :, 1:2] - x[:, :, :, 0:1],
                (x[:, :, :, 2:] - x[:, :, :, :-2]) / 2,
                x[:, :, :, -1:] - x[:, :, :, -2:-1],
            ],
            axis=axis,
        )

    msg = (
        f'_derivative received axis={axis}. This is meant to compute only '
        'temporal (axis=3) or spatial (axis=1/2) derivatives for tensors '
        'of shape (n_obs, spatial_1, spatial_2, temporal)'
    )
    raise ValueError(msg)


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
    result = tf.exp(
        -0.5
        * tf.reduce_sum((tf.expand_dims(x1, axis=1) - x2) ** 2, axis=-1)
        / sigma**2
    )
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
        return tf.reduce_mean(1 - tf.exp(-((x1 - x2) ** 2)))


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


class MaterialDerivativeLoss(tf.keras.losses.Loss):
    """Loss class for the material derivative. This is the left hand side of
    the Navier-Stokes equation and is equal to internal + external forces
    divided by density.

    References
    ----------
    https://en.wikipedia.org/wiki/Material_derivative
    """

    LOSS_METRIC = MeanAbsoluteError()

    def _compute_md(self, x, fidx):
        """Compute material derivative the feature given by the index fidx.
        It is assumed that for a given feature index fidx there is a pair of
        wind components u/v given by 2 * (fidx // 2) and 2 * (fidx // 2) + 1

        Parameters
        ----------
        x : tf.tensor
            synthetic output or high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)
        fidx : int
            Feature index to compute material derivative for.
        """
        uidx = 2 * (fidx // 2)
        vidx = 2 * (fidx // 2) + 1
        # df/dt
        x_div = _derivative(x[..., fidx], axis=3)
        # u * df/dx
        x_div += tf.math.multiply(
            x[..., uidx], _derivative(x[..., fidx], axis=1)
        )
        # v * df/dy
        x_div += tf.math.multiply(
            x[..., vidx], _derivative(x[..., fidx], axis=2)
        )

        return x_div

    def __call__(self, x1, x2):
        """Custom content loss that encourages accuracy of the material
        derivative. This assumes that the first 2 * N features are N u/v
        wind components at different hub heights and that the total number of
        features is either 2 * N or 2 * N + 1

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
        hub_heights = x1.shape[-1] // 2

        msg = (
            f'The {self.__class__.__name__} is meant to be used on '
            'spatiotemporal data only. Received tensor(s) that are not 5D'
        )
        assert len(x1.shape) == 5 and len(x2.shape) == 5, msg

        x1_div = tf.stack(
            [
                self._compute_md(x1, fidx=i)
                for i in range(0, 2 * hub_heights, 2)
            ]
        )
        x2_div = tf.stack(
            [
                self._compute_md(x2, fidx=i)
                for i in range(0, 2 * hub_heights, 2)
            ]
        )

        return self.LOSS_METRIC(x1_div, x2_div)


class SpatialDerivativeLoss(tf.keras.losses.Loss):
    """Loss class to encourage accurary of spatial derivatives."""

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Custom content loss that encourages accuracy of spatial derivatives

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
        msg = (
            f'The {self.__class__.__name__} is meant to be used on spatial or '
            'spatiotemporal data only. Received tensor(s) that are not at '
            'least 4D'
        )
        assert len(x1.shape) >= 4 and len(x2.shape) >= 4, msg

        x1_div = _derivative(x1, axis=1) + _derivative(x1, axis=2)
        x2_div = _derivative(x2, axis=1) + _derivative(x2, axis=2)

        return self.LOSS_METRIC(x1_div, x2_div)


class TemporalDerivativeLoss(tf.keras.losses.Loss):
    """Loss class to encourage accurary of temporal derivative."""

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Custom content loss that encourages accuracy of temporal derivative

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
        msg = (
            f'The {self.__class__.__name__} is meant to be used on '
            'spatiotemporal data only. Received tensor(s) that are not 5D'
        )
        assert len(x1.shape) == 5 and len(x2.shape) == 5, msg

        x1_div = _derivative(x1, axis=3)
        x2_div = _derivative(x2, axis=3)

        return self.LOSS_METRIC(x1_div, x2_div)


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


class SpatialExtremesLoss(tf.keras.losses.Loss):
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


class TemporalExtremesLoss(tf.keras.losses.Loss):
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


class SpatialFftLoss(tf.keras.losses.Loss):
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


class SpatiotemporalFftLoss(tf.keras.losses.Loss):
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


class LowResLoss(tf.keras.losses.Loss):
    """Content loss that is calculated by coarsening the synthetic and true
    high-resolution data pairs and then performing the pointwise content loss
    on the low-resolution fields"""

    EX_LOSS_METRICS: ClassVar = {
        'SpatialExtremesLoss': SpatialExtremesLoss,
        'TemporalExtremesLoss': TemporalExtremesLoss,
    }

    def __init__(
        self,
        s_enhance=1,
        t_enhance=1,
        t_method='average',
        tf_loss='MeanSquaredError',
        ex_loss=None,
    ):
        """Initialize the loss with given weight

        Parameters
        ----------
        s_enhance : int
            factor by which to coarsen spatial dimensions. 1 will keep the
            spatial axes as high-res
        t_enhance : int
            factor by which to coarsen temporal dimension. 1 will keep the
            temporal axes as high-res
        t_method : str
            Accepted options: [subsample, average]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps
        tf_loss : str
            The tensorflow loss function to operate on the low-res fields. Must
            be the name of a loss class that can be retrieved from
            ``tf.keras.losses`` e.g., "MeanSquaredError" or "MeanAbsoluteError"
        ex_loss : None | str
            Optional additional loss metric evaluating the spatial or temporal
            extremes of the high-res data. Can be "SpatialExtremesLoss" or
            "TemporalExtremesLoss" (keys in ``EX_LOSS_METRICS``).
        """

        super().__init__()
        self._s_enhance = s_enhance
        self._t_enhance = t_enhance
        self._t_method = str(t_method).casefold()
        self._tf_loss = getattr(tf.keras.losses, tf_loss)()
        self._ex_loss = ex_loss
        if self._ex_loss is not None:
            self._ex_loss = self.EX_LOSS_METRICS[self._ex_loss]()

    def _s_coarsen_4d_tensor(self, tensor):
        """Perform spatial coarsening on a 4D tensor of shape
        (n_obs, spatial_1, spatial_2, features)"""
        shape = tensor.shape
        tensor = tf.reshape(
            tensor,
            (
                shape[0],
                shape[1] // self._s_enhance,
                self._s_enhance,
                shape[2] // self._s_enhance,
                self._s_enhance,
                shape[3],
            ),
        )
        tensor = tf.math.reduce_sum(tensor, axis=(2, 4)) / self._s_enhance**2
        return tensor

    def _s_coarsen_5d_tensor(self, tensor):
        """Perform spatial coarsening on a 5D tensor of shape
        (n_obs, spatial_1, spatial_2, time, features)"""
        shape = tensor.shape
        tensor = tf.reshape(
            tensor,
            (
                shape[0],
                shape[1] // self._s_enhance,
                self._s_enhance,
                shape[2] // self._s_enhance,
                self._s_enhance,
                shape[3],
                shape[4],
            ),
        )
        tensor = tf.math.reduce_sum(tensor, axis=(2, 4)) / self._s_enhance**2
        return tensor

    def _t_coarsen_sample(self, tensor):
        """Perform temporal subsampling on a 5D tensor of shape
        (n_obs, spatial_1, spatial_2, time, features)"""
        assert len(tensor.shape) == 5
        tensor = tensor[:, :, :, :: self._t_enhance, :]
        return tensor

    def _t_coarsen_avg(self, tensor):
        """Perform temporal coarsening on a 5D tensor of shape
        (n_obs, spatial_1, spatial_2, time, features)"""
        shape = tensor.shape
        assert len(shape) == 5
        tensor = tf.reshape(
            tensor,
            (shape[0], shape[1], shape[2], -1, self._t_enhance, shape[4]),
        )
        tensor = tf.math.reduce_sum(tensor, axis=4) / self._t_enhance
        return tensor

    def __call__(self, x1, x2):
        """Custom content loss calculated on re-coarsened low-res fields

        Parameters
        ----------
        x1 : tf.tensor
            Synthetic high-res generator output, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            True high resolution data, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """

        assert x1.shape == x2.shape
        s_only = len(x1.shape) == 4

        ex_loss = tf.constant(0, dtype=x1.dtype)
        if self._ex_loss is not None:
            ex_loss = self._ex_loss(x1, x2)

        if self._s_enhance > 1 and s_only:
            x1 = self._s_coarsen_4d_tensor(x1)
            x2 = self._s_coarsen_4d_tensor(x2)

        elif self._s_enhance > 1 and not s_only:
            x1 = self._s_coarsen_5d_tensor(x1)
            x2 = self._s_coarsen_5d_tensor(x2)

        if self._t_enhance > 1 and self._t_method == 'average':
            x1 = self._t_coarsen_avg(x1)
            x2 = self._t_coarsen_avg(x2)

        if self._t_enhance > 1 and self._t_method == 'subsample':
            x1 = self._t_coarsen_sample(x1)
            x2 = self._t_coarsen_sample(x2)

        return self._tf_loss(x1, x2) + ex_loss


class PerceptualLoss(tf.keras.losses.Loss):
    """Perceptual loss that is calculated as MSE between feature maps of
    ground truth and synthetic data"""

    def __init__(self, layer_names=None):
        """
        Parameters
        ----------
        layer_names : list | None
            List of layer names in VGG16 to use to extract feature maps from
            ground truth and synthetic data. Defaults to ['block1_conv2',
            'block2_conv2']
        """
        super().__init__()
        # VGG16 for perceptual loss
        vgg = VGG16(weights='imagenet', include_top=False)
        vgg.trainable = False
        self.layer_names = layer_names
        if self.layer_names is None:
            self.layer_names = ['block1_conv2', 'block2_conv2']
        vgg_outputs = [vgg.get_layer(name).output for name in self.layer_names]
        self.feature_extractor = tf.keras.Model(
            inputs=vgg.input, outputs=vgg_outputs
        )

    def _feature_loss(self, x1, x2):
        """Calculate loss for a single feature. e.g. A single pair of tensors
        each with only 3 channels"""
        x1 = preprocess_input(x1)
        x2 = preprocess_input(x2)
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        if len(self.layer_names) == 1:
            x1 = [x1]
            x2 = [x2]
        loss = 0
        for x1_f, x2_f in zip(x1, x2):
            loss += tf.reduce_mean(tf.square(x1_f - x2_f))
        return loss

    def __call__(self, x1, x2):
        """Perceptual loss calculated on true and synthetic feature maps

        Parameters
        ----------
        x1 : tf.tensor
            Synthetic high-res generator output, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            True high resolution data, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """
        if len(x1.shape) == 5:
            new_shape = (
                x1.shape[0] * x1.shape[3],
                x1.shape[1],
                x1.shape[2],
                x1.shape[-1],
            )
            x1 = tf.reshape(x1, new_shape)
            x2 = tf.reshape(x2, new_shape)

        losses = []
        for i in range(x1.shape[-1]):
            x1_f = x1[..., i]
            x2_f = x2[..., i]

            # VGG input needs 3 RGB channels
            x1_f = tf.stack([x1_f] * 3, axis=-1)
            x2_f = tf.stack([x2_f] * 3, axis=-1)

            losses.append(self._feature_loss(x1_f, x2_f))

        return tf.reduce_mean(losses)
