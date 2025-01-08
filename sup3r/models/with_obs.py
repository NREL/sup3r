"""Sup3r model with training on observation data."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError

from .base import Sup3rGan

logger = logging.getLogger(__name__)


class Sup3rGanWithObs(Sup3rGan):
    """Sup3r GAN model which incorporates observation data into content loss.
    """

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
        val_exo_data = self.get_high_res_exo_input(batch.high_res)
        high_res_gen = self._tf_generate(batch.low_res, val_exo_data)
        _, v_loss_details = self.calc_loss(
            batch.high_res,
            high_res_gen,
            weight_gen_advers=weight_gen_advers,
            train_gen=False,
            train_disc=False,
        )
        v_loss_details['loss_obs'] = self.cal_loss_obs(batch.obs, high_res_gen)

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
            *loss_out, hi_res_gen = self._get_hr_exo_and_loss(
                low_res, hi_res_true, **calc_loss_kwargs
            )
            loss, loss_details = loss_out
            loss_obs = self.calc_loss_obs(obs_data, hi_res_gen)
            if not tf.reduce_any(tf.math.is_nan(loss_obs)):
                loss += loss_obs
                loss_details.update({'loss_obs': loss_obs})
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
        obs_loss = tf.constant(np.nan)
        if obs_data is not None:
            mask = tf.math.is_nan(obs_data)
            masked_obs = obs_data[~mask]
            if len(masked_obs) > 0:
                obs_loss = MeanAbsoluteError()(
                    masked_obs,
                    hi_res_gen[..., : len(self.hr_out_features)][~mask],
                )
        return obs_loss
