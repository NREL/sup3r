# -*- coding: utf-8 -*-
"""Wind super resolution GAN with handling of low and high res topography
inputs."""
import numpy as np
import logging
import tensorflow as tf
from phygnn.layers.custom_layers import Sup3rAdder, Sup3rConcat

from sup3r.models.base import Sup3rGan


logger = logging.getLogger(__name__)


class WindGan(Sup3rGan):
    """Wind super resolution GAN with handling of low and high res topography
    inputs.

    Modifications to standard Sup3rGan:
        - Hi res topography is expected as the last feature channel in the true
          data in the true batch observation. This topo channel is appended to
          the generated output so the discriminator can look at the wind fields
          compared to the associated hi res topo.
        - If a custom Sup3rAdder or Sup3rConcat layer (from phygnn) is present
          in the network, the hi-res topography will be added or concatenated
          to the data at that point in the network during either training or
          the forward pass.
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
            phygnn Sup3rAdder or Sup3rConcat layer if found in the generative
            network. This differs from the exogenous_data input in that
            exogenous_data always matches the low-res input. For this function,
            hi_res_topo can also be a 2D array (spatial_1, spatial_2). Note
            that this input gets normalized if norm_in=True.
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

        if norm_in and self._means is not None:
            idf = self.training_features.index('topography')
            hi_res_topo = ((hi_res_topo.copy() - self._means[idf])
                           / self._stdevs[idf])

        if len(hi_res_topo.shape) > 2:
            slicer = [0] * len(hi_res_topo.shape)
            slicer[1] = slice(None)
            slicer[2] = slice(None)
            hi_res_topo = hi_res_topo[tuple(slicer)]

        if len(hi_res.shape) == 4:
            hi_res_topo = np.expand_dims(hi_res_topo, axis=(0, 3))
            hi_res_topo = np.repeat(hi_res_topo, hi_res.shape[0], axis=0)
        elif len(hi_res.shape) == 5:
            hi_res_topo = np.expand_dims(hi_res_topo, axis=(0, 3, 4))
            hi_res_topo = np.repeat(hi_res_topo, hi_res.shape[0], axis=0)
            hi_res_topo = np.repeat(hi_res_topo, hi_res.shape[3], axis=3)

        if len(hi_res_topo.shape) != len(hi_res.shape):
            msg = ('hi_res and hi_res_topo arrays are not of the same rank: '
                   '{} and {}'.format(hi_res.shape, hi_res_topo.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        return hi_res_topo

    def generate(self, low_res, norm_in=True, un_norm_out=True,
                 exogenous_data=None):
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
        exogenous_data : ndarray | list | None
            Exogenous data for topography inputs. The first entry in this list
            (or only entry) is a low-resolution topography array that can be
            concatenated to the low_res input array. The second entry is
            high-resolution topography (either 2D or 4D/5D depending on if
            spatial or spatiotemporal super res).

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """

        low_res_topo = None
        hi_res_topo = None
        if isinstance(exogenous_data, np.ndarray):
            low_res_topo = exogenous_data
        elif isinstance(exogenous_data, (list, tuple)):
            low_res_topo = exogenous_data[0]
            if len(exogenous_data) > 1:
                hi_res_topo = exogenous_data[1]

        exo_check = (low_res is None or not self._needs_lr_exo(low_res))
        low_res = (low_res if exo_check
                   else np.concatenate((low_res, low_res_topo), axis=-1))

        if norm_in and self._means is not None:
            low_res = self.norm_input(low_res)

        hi_res = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
                if (isinstance(layer, (Sup3rAdder, Sup3rConcat))
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
            phygnn Sup3rAdder or Sup3rConcat layer if found in the generative
            network. This differs from the exogenous_data input in that
            exogenous_data always matches the low-res input.

        Returns
        -------
        hi_res : tf.Tensor
            Synthetically generated high-resolution data
        """

        hi_res = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
                if (isinstance(layer, (Sup3rAdder, Sup3rConcat))
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

    def calc_val_loss(self, batch_handler, weight_gen_advers, loss_details):
        """Calculate the validation loss at the current state of model training

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        loss_details : dict
            Namespace of the breakdown of loss components

        Returns
        -------
        loss_details : dict
            Same as input but now includes val_* loss info
        """
        logger.debug('Starting end-of-epoch validation loss calculation...')
        loss_details['n_obs'] = 0
        for val_batch in batch_handler.val_data:
            high_res_gen = self._tf_generate(val_batch.low_res,
                                             val_batch.high_res[..., -1:])
            _, v_loss_details = self.calc_loss(
                val_batch.high_res, high_res_gen,
                weight_gen_advers=weight_gen_advers,
                train_gen=False, train_disc=False)

            loss_details = self.update_loss_details(loss_details,
                                                    v_loss_details,
                                                    len(val_batch),
                                                    prefix='val_')
        return loss_details
