# -*- coding: utf-8 -*-
"""Wind conditional moment estimator with handling of low and
high res topography inputs."""
import logging
import tensorflow as tf

from sup3r.models.abstract import AbstractWindInterface
from sup3r.models.conditional_moments import Sup3rCondMom


logger = logging.getLogger(__name__)


class WindCondMom(Sup3rCondMom, AbstractWindInterface):
    """Wind conditional moment estimator with handling of low and
    high res topography inputs.

    Modifications to standard Sup3rCondMom:
        - Hi res topography is expected as the last feature channel in the true
          data in the true batch observation.
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
        AbstractWindInterface.set_model_params_wind(**kwargs)
        super().set_model_params(**kwargs)

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
        return self.generate_wind(low_res, norm_in,
                                  un_norm_out, exogenous_data)

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
        return self._tf_generate_wind(low_res, hi_res_topo)

    @tf.function
    def calc_loss(self, hi_res_true, hi_res_gen, mask, **kwargs):
        """Calculate the loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.
        mask : tf.Tensor
            Mask to apply
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

        return super().calc_loss(hi_res_true, hi_res_gen, mask, **kwargs)

    def run_gradient_descent(self, low_res, hi_res_true, mask,
                             training_weights,
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
        mask : np.ndarray
            Mask of high-resolution data in a 4D or 5D array:
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
            loss_out = self.calc_loss(hi_res_true, hi_res_gen, mask,
                                      **calc_loss_kwargs)
            loss, loss_details = loss_out

            grad = tape.gradient(loss, training_weights)

        if optimizer is None:
            optimizer = self.optimizer

        optimizer.apply_gradients(zip(grad, training_weights))

        return loss_details

    def calc_val_loss(self, batch_handler, loss_details):
        """Calculate the validation loss at the current state of model training

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
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
                val_batch.output, high_res_gen, val_batch.mask)

            loss_details = self.update_loss_details(loss_details,
                                                    v_loss_details,
                                                    len(val_batch),
                                                    prefix='val_')
        return loss_details