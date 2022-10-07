# -*- coding: utf-8 -*-
"""Sup3r model software"""
import numpy as np
import logging
import tensorflow as tf
from phygnn.layers.custom_layers import Sup3rWindTopo

from sup3r.models.base import Sup3rGan


logger = logging.getLogger(__name__)


class WindCC(Sup3rGan):
    """Wind climate change model.

    Modifications to standard Sup3rGan:
        - Hi res topography is expected as the last feature channel in the true
          data in the true batch observation. This topo channel is appended to
          the generated output so the discriminator can look at the wind fields
          compared to the associated hi res topo.
    """

    def set_model_params(self, **kwargs):
        """Set parameters used for training the model

        Parameters
        ----------
        kwargs : dict
            Keyword arguments including 'training_features', 'output_features',
            'smoothed_features', 's_enhance', 't_enhance', 'smoothing'
        """
        output_features = kwargs['output_features']
        msg = ('Last output feature from the data handler must be topography '
               'to train the WindCC model, but received output features: {}'
               .format(output_features))
        assert output_features[-1] == 'topography', msg
        output_features.remove('topography')
        kwargs['output_features'] = output_features
        super().set_model_params(**kwargs)

    def _reshape_norm_topo(self, hi_res, hi_res_topo, norm_in=True):
        """Reshape the hi_res_topo to match the hi_res tensor (if necessary)
        and normalize (if requested).

        Parameters
        ----------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        hi_res_topo : np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features) corresponding to the high-resolution
            spatial_1 and spatial_2. This data will be input to the custom
            phygnn Sup3rWindTopo layer if found in the generative network. This
            differs from the exogenous_data input in that exogenous_data always
            matches the low-res input. For this function, hi_res_topo can also
            be a 2D array (spatial_1, spatial_2). Note that this input gets
            normalized if norm_in=True.
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1. This also normalizes
            hi_res_topo.

        Returns
        -------
        hi_res_topo : np.ndarray
            Same as input but reshaped to match hi_res (if necessary) and
            normalized (if requested)
        """
        if hi_res_topo is None:
            return hi_res_topo

        if norm_in:
            idf = self.training_features.index('topography')
            hi_res_topo -= self._means[idf]
            hi_res_topo /= self._stdevs[idf]

        if len(hi_res_topo.shape) == 2 and len(hi_res.shape) == 4:
            hi_res_topo = np.expand_dims(hi_res_topo, axis=(0, 3))

        elif len(hi_res_topo.shape) == 2 and len(hi_res.shape) == 5:
            hi_res_topo = np.expand_dims(hi_res_topo, axis=(0, 3, 4))

        return hi_res_topo

    def generate(self, low_res, norm_in=True, un_norm_out=True,
                 exogenous_data=None, hi_res_topo=None):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1. This also normalizes
            hi_res_topo.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units
        exogenous_data : ndarray | None
            Exogenous data array, usually a 4D or 5D array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        hi_res_topo : np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features) corresponding to the high-resolution
            spatial_1 and spatial_2. This data will be input to the custom
            phygnn Sup3rWindTopo layer if found in the generative network. This
            differs from the exogenous_data input in that exogenous_data always
            matches the low-res input. For this function, hi_res_topo can also
            be a 2D array (spatial_1, spatial_2). Note that this input gets
            normalized if norm_in=True.

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        low_res = (low_res if not exogenous_data
                   else np.concatenate((low_res, exogenous_data), axis=-1))

        if norm_in and self._means is not None:
            low_res = self.norm_input(low_res)

        hi_res = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
                if (isinstance(layer, Sup3rWindTopo)
                        and hi_res_topo is not None):
                    hi_res_topo = self._reshape_norm_topo(hi_res, hi_res_topo,
                                                          norm_in=norm_in)
                    hi_res = layer(hi_res, hi_res_topo)
                else:
                    hi_res = layer(hi_res)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, hi_res.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        hi_res = hi_res.numpy()

        if un_norm_out and self._means is not None:
            hi_res = self.un_norm_output(hi_res)

        return hi_res

    @tf.function
    def _tf_generate(self, low_res, hi_res_topo):
        """Use the generator model to generate high res data from los res input

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data. The generator should always
            received normalized data with mean=0 stdev=1.
        hi_res_topo : np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features) corresponding to the high-resolution
            spatial_1 and spatial_2. This data will be input to the custom
            phygnn Sup3rWindTopo layer if found in the generative network. This
            differs from the exogenous_data input in that exogenous_data always
            matches the low-res input.

        Returns
        -------
        hi_res : tf.Tensor
            Synthetically generated high-resolution data
        """

        hi_res = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
                if (isinstance(layer, Sup3rWindTopo)
                        and hi_res_topo is not None):
                    hi_res = layer(hi_res, hi_res_topo)
                else:
                    hi_res = layer(hi_res)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, hi_res.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        return hi_res

    @tf.function
    def calc_loss(self, hi_res_true, hi_res_gen, **kwargs):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.
        kwargs : dict
            Key word arguments for:
            Sup3rGan.calc_loss(hi_res_true, hi_res_gen, **kwargs)

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the network being trained
            (either generator or one of the discriminators)
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        # append the true topography to the generated synthetic wind data
        hi_res_gen = tf.concat((hi_res_gen, hi_res_true[..., -1:]), axis=-1)

        return super().calc_loss(hi_res_true, hi_res_gen, **kwargs)

    def run_gradient_descent(self, low_res, hi_res_true, training_weights,
                             optimizer=None, **calc_loss_kwargs):
        """Run gradient descent for one mini-batch of (low_res, hi_res_true)
        and adjust NN weights

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data in a 4D or 5D array:
            (n_observations, spatial_1, spatial_2, features)
            (n_observations, spatial_1, spatial_2, temporal, features)
        hi_res_true : np.ndarray
            Real high-resolution data in a 4D or 5D array:
            (n_observations, spatial_1, spatial_2, features)
            (n_observations, spatial_1, spatial_2, temporal, features)
        training_weights : list
            A list of layer weights that are to-be-trained based on the
            current loss weight values.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer class to use to update weights. This can be different if
            you're training just the generator or one of the discriminator
            models. Defaults to the generator optimizer.
        calc_loss_kwargs : dict
            Kwargs to pass to the self.calc_loss() method

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        hi_res_topo = hi_res_true[..., -1:]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(training_weights)

            hi_res_gen = self._tf_generate(low_res, hi_res_topo)
            loss_out = self.calc_loss(hi_res_true, hi_res_gen,
                                      **calc_loss_kwargs)
            loss, loss_details = loss_out

            grad = tape.gradient(loss, training_weights)

        if optimizer is None:
            optimizer = self.optimizer

        optimizer.apply_gradients(zip(grad, training_weights))

        return loss_details
