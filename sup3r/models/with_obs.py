"""Sup3r model with training on observation data."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from phygnn.layers.custom_layers import (
    Sup3rConcatObs,
    Sup3rConcatObsBlock,
    Sup3rImpute,
)
from tensorflow.keras.losses import MeanAbsoluteError

from sup3r.utilities.utilities import RANDOM_GENERATOR

from .base import Sup3rGan

logger = logging.getLogger(__name__)


class Sup3rGanWithObs(Sup3rGan):
    """Sup3r GAN model with additional observation data content loss. This
    model is useful for when observations are available for the training domain
    but not for the production domain."""

    def _calc_val_loss(self, batch, weight_gen_advers, loss_details):
        """Calculate the validation loss at the current state of model training
        for a given batch

        Parameters
        ----------
        batch : DsetTuple
            Object with ``.high_res``, ``.low_res``, and ``.obs`` arrays
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        loss_details : dict
            Namespace of the breakdown of loss components

        Returns
        -------
        loss_details : dict
            Same as input with updated val_* loss info
        """
        _, v_loss_details, hi_res_gen, _ = self._get_hr_exo_and_loss(
            batch.low_res,
            batch.high_res,
            weight_gen_advers=weight_gen_advers,
            train_gen=False,
            train_disc=False,
        )

        v_loss_details['loss_obs'] = self.calc_loss_obs(batch.obs, hi_res_gen)

        loss_details = self.update_loss_details(
            loss_details, v_loss_details, len(batch), prefix='val_'
        )
        return loss_details

    def _get_batch_loss_details(
        self,
        batch,
        train_gen,
        only_gen,
        gen_too_good,
        train_disc,
        only_disc,
        disc_too_good,
        weight_gen_advers,
        multi_gpu=False,
    ):
        """Get loss details for a given batch for the current epoch.

        Parameters
        ----------
        batch : sup3r.preprocessing.base.DsetTuple
            Object with ``.low_res``, ``.high_res``, and ``.obs`` arrays
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        only_gen : bool
            Flag whether to only train the generator for this set of epochs
        gen_too_good : bool
            Flag whether to skip training the generator and only train the
            discriminator, due to superior performance, for this batch.
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
        only_disc : bool
            Flag whether to only train the discriminator for this set of epochs
        gen_too_good : bool
            Flag whether to skip training the discriminator and only train the
            generator, due to superior performance, for this batch.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        multi_gpu : bool
            Flag to break up the batch for parallel gradient descent
            calculations on multiple gpus. If True and multiple GPUs are
            present, each batch from the batch_handler will be divided up
            between the GPUs and resulting gradients from each GPU will be
            summed and then applied once per batch at the nominal learning
            rate that the model and optimizer were initialized with.
            If true and multiple gpus are found, ``default_device`` device
            should be set to /gpu:0

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components for the given batch
        """
        trained_gen = False
        trained_disc = False
        if only_gen or (train_gen and not gen_too_good):
            trained_gen = True
            b_loss_details = self.timer(self.run_gradient_descent)(
                batch.low_res,
                batch.high_res,
                self.generator_weights,
                obs_data=getattr(batch, 'obs', None),
                weight_gen_advers=weight_gen_advers,
                optimizer=self.optimizer,
                train_gen=True,
                train_disc=False,
                multi_gpu=multi_gpu,
            )

        if only_disc or (train_disc and not disc_too_good):
            trained_disc = True
            b_loss_details = self.timer(self.run_gradient_descent)(
                batch.low_res,
                batch.high_res,
                self.discriminator_weights,
                weight_gen_advers=weight_gen_advers,
                optimizer=self.optimizer_disc,
                train_gen=False,
                train_disc=True,
                multi_gpu=multi_gpu,
            )

        b_loss_details['gen_trained_frac'] = float(trained_gen)
        b_loss_details['disc_trained_frac'] = float(trained_disc)

        if 'loss_obs' in b_loss_details and not tf.math.is_nan(
            b_loss_details['loss_obs']
        ):
            loss_update = b_loss_details['loss_gen']
            loss_update += b_loss_details['loss_obs']
            b_loss_details.update({'loss_gen': loss_update})
        return b_loss_details

    def _get_parallel_grad(
        self,
        low_res,
        hi_res_true,
        training_weights,
        obs_data=None,
        **calc_loss_kwargs,
    ):
        """Compute gradient for one mini-batch of (low_res, hi_res_true,
        obs_data) across multiple GPUs. Can include observation data as well.
        """

        futures = []
        start_time = time.time()
        lr_chunks = np.array_split(low_res, len(self.gpu_list))
        hr_true_chunks = np.array_split(hi_res_true, len(self.gpu_list))
        obs_data_chunks = (
            [None] * len(hr_true_chunks)
            if obs_data is None
            else np.array_split(obs_data, len(self.gpu_list))
        )
        split_mask = False
        mask_chunks = None
        if 'mask' in calc_loss_kwargs:
            split_mask = True
            mask_chunks = np.array_split(
                calc_loss_kwargs['mask'], len(self.gpu_list)
            )

        with ThreadPoolExecutor(max_workers=len(self.gpu_list)) as exe:
            for i in range(len(self.gpu_list)):
                if split_mask:
                    calc_loss_kwargs['mask'] = mask_chunks[i]
                futures.append(
                    exe.submit(
                        self.get_single_grad,
                        lr_chunks[i],
                        hr_true_chunks[i],
                        training_weights,
                        obs_data=obs_data_chunks[i],
                        device_name=f'/gpu:{i}',
                        **calc_loss_kwargs,
                    )
                )

        return self._sum_parallel_grad(futures, start_time=start_time)

    def run_gradient_descent(
        self,
        low_res,
        hi_res_true,
        training_weights,
        obs_data=None,
        optimizer=None,
        multi_gpu=False,
        **calc_loss_kwargs,
    ):
        """Run gradient descent for one mini-batch of (low_res, hi_res_true)
        and update weights

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
        obs_data : tf.Tensor | None
            Optional observation data to use in additional content loss term.
            This needs to have NaNs where there is no observation data.
            (n_observations, spatial_1, spatial_2, features)
            (n_observations, spatial_1, spatial_2, temporal, features)
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer class to use to update weights. This can be different if
            you're training just the generator or one of the discriminator
            models. Defaults to the generator optimizer.
        multi_gpu : bool
            Flag to break up the batch for parallel gradient descent
            calculations on multiple gpus. If True and multiple GPUs are
            present, each batch from the batch_handler will be divided up
            between the GPUs and resulting gradients from each GPU will be
            summed and then applied once per batch at the nominal learning
            rate that the model and optimizer were initialized with.
        calc_loss_kwargs : dict
            Kwargs to pass to the self.calc_loss() method

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        self.timer.start()
        if optimizer is None:
            optimizer = self.optimizer

        if not multi_gpu or len(self.gpu_list) < 2:
            grad, loss_details = self.get_single_grad(
                low_res,
                hi_res_true,
                training_weights,
                obs_data=obs_data,
                device_name=self.default_device,
                **calc_loss_kwargs,
            )
            optimizer.apply_gradients(zip(grad, training_weights))
            self.timer.stop()
            logger.debug(
                'Finished single gradient descent step in %s',
                self.timer.elapsed_str,
            )
        else:
            total_grad, loss_details = self._get_parallel_grad(
                low_res,
                hi_res_true,
                training_weights,
                obs_data,
                **calc_loss_kwargs,
            )
            optimizer.apply_gradients(zip(total_grad, training_weights))

        return loss_details

    @tf.function
    def get_single_grad(
        self,
        low_res,
        hi_res_true,
        training_weights,
        obs_data=None,
        device_name=None,
        **calc_loss_kwargs,
    ):
        """Run gradient descent for one mini-batch of (low_res, hi_res_true),
        do not update weights, just return gradient details.

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
        obs_data : tf.Tensor | None
            Optional observation data to use in additional content loss term.
            This needs to have NaNs where there is no observation data.
            (n_observations, spatial_1, spatial_2, features)
            (n_observations, spatial_1, spatial_2, temporal, features)
        device_name : None | str
            Optional tensorflow device name for GPU placement. Note that if a
            GPU is available, variables will be placed on that GPU even if
            device_name=None.
        calc_loss_kwargs : dict
            Kwargs to pass to the self.calc_loss() method

        Returns
        -------
        grad : list
            a list or nested structure of Tensors (or IndexedSlices, or None,
            or CompositeTensor) representing the gradients for the
            training_weights
        loss_details : dict
            Namespace of the breakdown of loss components
        """
        with tf.device(device_name), tf.GradientTape(
            watch_accessed_variables=False
        ) as tape:
            tape.watch(training_weights)
            loss, loss_details, hi_res_gen, _ = self._get_hr_exo_and_loss(
                low_res, hi_res_true, **calc_loss_kwargs
            )
            loss_obs = self.calc_loss_obs(obs_data, hi_res_gen)
            loss_update = {'loss_obs': loss_obs}
            if calc_loss_kwargs['train_gen'] and not tf.reduce_any(
                tf.math.is_nan(loss_obs)
            ):
                loss += loss_obs
                loss_update['loss_gen'] = loss
                loss_details.update(loss_update)
            grad = tape.gradient(loss, training_weights)
        return grad, loss_details

    @tf.function
    def calc_loss_obs(self, obs_data, hi_res_gen):
        """Calculate loss term for the observation data vs generated
        high-resolution data

        Parameters
        ----------
        obs_data : tf.Tensor | None
            Observation data to use in additional content loss term.
            This needs to have NaNs where there is no observation data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.

        Returns
        -------
        loss : tf.Tensor
            0D tensor of observation loss
        """
        loss_obs = tf.constant(np.nan)
        if obs_data is not None:
            mask = tf.math.is_nan(obs_data)
            masked_obs = obs_data[~mask]
            if len(masked_obs) > 0:
                loss_obs = MeanAbsoluteError()(
                    masked_obs,
                    hi_res_gen[..., : len(self.hr_out_features)][~mask],
                )
        return loss_obs


class Sup3rGanFixedObs(Sup3rGan):
    """Sup3r GAN model which includes mid network observation fixing. This
    model is useful for when production runs will be over a domain for which
    observation data is available."""

    def __init__(self, *args, obs_frac=None, **kwargs):
        """
        Initialize the Sup3rGanFixedObs model.

        Parameters
        ----------
        args : list
            Positional args for ``Sup3rGan`` parent class.
        obs_frac : dict
            Fraction of the batch that should be "fixed" with observations.
            Should include ``spatial`` key and optionally ``time`` key if this
            is a spatiotemporal model. The values should correspond roughly to
            the fraction of the production domain for which observations are
            available (spatial) and the fraction of the full time period that
            these cover. For each batch a spatial frac will be selected by
            uniformly selecting from the range ``(0, obs_frac['spatial'])``
        kwargs : dict
            Keyword arguments for the ``Sup3rGan`` parent class.
        """
        self.obs_frac = {} if obs_frac is None else obs_frac
        super().__init__(*args, **kwargs)

    @property
    def obs_features(self):
        """Get list of exogenous observation feature names the model uses.
        These come from the names of the ``Sup3rObs`` layers."""
        # pylint: disable=E1101
        features = []
        if hasattr(self, '_gen'):
            for layer in self._gen.layers:
                check = isinstance(
                    layer, (Sup3rConcatObs, Sup3rConcatObsBlock, Sup3rImpute)
                )
                check = check and layer.name not in features
                if check:
                    features.append(layer.name)
        return features

    def _get_loss_obs_comparison(self, hi_res_true, hi_res_gen, obs_mask):
        """Get loss for observation locations and for non observation
        locations."""

        hr_true = [
            hi_res_true[..., self.hr_out_features.index(f)]
            for f in self.obs_features
        ]
        hr_true = tf.stack(hr_true, axis=-1)
        hr_gen = [
            hi_res_gen[..., self.hr_out_features.index(f)]
            for f in self.obs_features
        ]
        hr_gen = tf.stack(hr_gen, axis=-1)

        loss_obs = MeanAbsoluteError()(hr_true[~obs_mask], hr_gen[~obs_mask])
        loss_non_obs = MeanAbsoluteError()(hr_true[obs_mask], hr_gen[obs_mask])
        return loss_obs, loss_non_obs

    def _get_obs_mask(self, hi_res, spatial_frac, time_frac=None):
        """Define observation mask for the current batch. This is done
        with a spatial mask and a temporal mask since often observation data
        might be very sparse spatially but cover most of the full time period
        for those locations."""
        obs_mask = RANDOM_GENERATOR.choice(
            [True, False],
            size=hi_res.shape[1:3],
            p=[1 - spatial_frac, spatial_frac],
        )
        if self.is_5d:
            sp_mask = obs_mask.copy()
            obs_mask = RANDOM_GENERATOR.choice(
                [True, False],
                size=hi_res.shape[1:-1],
                p=[1 - time_frac, time_frac],
            )
            obs_mask[sp_mask] = True
        return np.repeat(obs_mask[None, ...], hi_res.shape[0], axis=0)

    def init_weights(self, lr_shape, hr_shape, device=None):
        """Initialize the generator and discriminator weights with device
        placement.

        Parameters
        ----------
        lr_shape : tuple
            Shape of one batch of low res input data for sup3r resolution. Note
            that the batch size (axis=0) must be included, but the actual batch
            size doesnt really matter.
        hr_shape : tuple
            Shape of one batch of high res input data for sup3r resolution.
            Note that the batch size (axis=0) must be included, but the actual
            batch size doesnt really matter.
        device : str | None
            Option to place model weights on a device. If None,
            self.default_device will be used.
        """

        if device is None:
            device = self.default_device

        logger.info('Initializing model weights on device "{}"'.format(device))
        low_res = np.ones(lr_shape).astype(np.float32)
        hi_res = np.ones(hr_shape).astype(np.float32)

        hr_exo_shape = hr_shape[:-1] + (1,)
        hr_exo = np.ones(hr_exo_shape).astype(np.float32)

        with tf.device(device):
            hr_exo_data = {}
            for feature in self.hr_exo_features + self.obs_features:
                hr_exo_data[feature] = hr_exo
            _ = self._tf_generate(low_res, hr_exo_data)
            _ = self._tf_discriminate(hi_res)

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        params = super().model_params
        params['obs_frac'] = self.obs_frac
        return params

    def get_high_res_exo_input(self, hi_res_true):
        """Mask high res data to act as sparse observation data. Add this to
        the standard high res exo input"""
        exo_data = super().get_high_res_exo_input(hi_res_true)
        spatial_frac = RANDOM_GENERATOR.uniform(0, self.obs_frac['spatial'])
        time_frac = self.obs_frac.get('time', None)
        obs_mask = self._get_obs_mask(hi_res_true, spatial_frac, time_frac)
        for feature in self.obs_features:
            # obs_features can include a _obs suffix to avoid conflict with
            # fully gridded exo features
            f_idx = self.hr_out_features.index(feature.replace('_obs', ''))
            exo_data[feature] = tf.where(
                obs_mask, np.nan, hi_res_true[..., f_idx]
            )[..., None]
        exo_data['mask'] = obs_mask
        return exo_data

    def _get_hr_exo_and_loss(
        self,
        low_res,
        hi_res_true,
        **calc_loss_kwargs,
    ):
        """Get high-resolution exogenous data, generate synthetic output, and
        compute loss. Includes artificially masking hi res data to act as
        sparse observation data."""
        out = super()._get_hr_exo_and_loss(
            low_res, hi_res_true, **calc_loss_kwargs
        )
        loss, loss_details, hi_res_gen, hi_res_exo = out
        loss_obs, loss_non_obs = self._get_loss_obs_comparison(
            hi_res_true,
            hi_res_gen,
            hi_res_exo['mask'],
        )
        loss_update = {
            'loss_obs': loss_obs,
            'loss_non_obs': loss_non_obs,
        }
        loss_details.update(loss_update)
        return loss, loss_details, hi_res_gen, hi_res_exo
