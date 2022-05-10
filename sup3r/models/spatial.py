# -*- coding: utf-8 -*-
"""Sup3r model software"""
import copy
import os
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers

from sup3r.models.base import BaseModel


logger = logging.getLogger(__name__)


class SpatialGan(BaseModel):
    """Spatial super resolution GAN model"""

    def __init__(self, gen_layers, disc_layers,
                 optimizer=None, learning_rate=1e-4,
                 optimizer_s=None, learning_rate_s=None,
                 history=None, version_record=None, meta=None,
                 means=None, stdevs=None, name=None):
        """
        Parameters
        ----------
        gen_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            generative super resolving model. Can also be a str filepath to a
            .json config file containing the input layers argument or a .pkl
            for a saved pre-trained model.
        disc_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative model. Can also be a str filepath to a
            .json config file containing the input layers argument or a .pkl
            for a saved pre-trained model.
        optimizer : tensorflow.keras.optimizers | dict | None
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        optimizer_s : tensorflow.keras.optimizers | dict | None
            Same as optimizer input, but if specified this makes a different
            optimizer just for the spatial discriminator model.
        learning_rate_s : float, optional
            Same as learning_rate input, but if specified this makes a
            different learning_rate just for the spatial discriminator model.
        history : pd.DataFrame | str | None
            Model training history with "epoch" index, str pointing to a saved
            history csv file with "epoch" as first column, or None for clean
            history
        version_record : dict | None
            Optional record of import package versions. None (default) will
            save active environment versions. A dictionary will be interpreted
            as versions from a loaded model and will be saved as an attribute.
        meta : dict | None
            Model meta data that describes how the model was created.
        means : np.ndarray | list | None
            Set of mean values for data normalization with the same length as
            number of features. Can be used to maintain a consistent
            normalization scheme between transfer learning domains.
        stdevs : np.ndarray | list | None
            Set of stdev values for data normalization with the same length as
            number of features. Can be used to maintain a consistent
            normalization scheme between transfer learning domains.
        name : str | None
            Optional name for the GAN.
        """

        super().__init__(optimizer=optimizer,
                         learning_rate=learning_rate,
                         history=history, version_record=version_record,
                         meta=meta, means=means, stdevs=stdevs, name=name)

        self._gen = self.load_network(gen_layers, 'generator')
        self._disc = self.load_network(disc_layers, 'discriminator')

        optimizer_s = optimizer_s or copy.deepcopy(optimizer)
        learning_rate_s = learning_rate_s or learning_rate
        self._optimizer_s = self.init_optimizer(optimizer_s, learning_rate_s)

    def save(self, out_dir):
        """Save the GAN with its sub-networks to a directory.

        Parameters
        ----------
        out_dir : str
            Directory to save GAN model files. This directory will be created
            if it does not already exist.
        """

        fp_disc = os.path.join(out_dir, 'model_disc.pkl')
        self.disc.save(fp_disc)
        super().save(out_dir)
        logger.info('Saved GAN to disk in directory: {}'.format(out_dir))

    @classmethod
    def load(cls, out_dir):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        out_dir : str
            Directory to load GAN model files from.

        Returns
        -------
        out : SpatialGan
            Returns a pretrained SpatialGan model that was previously saved to
            out_dir
        """

        logger.info('Loading GAN from disk in directory: {}'.format(out_dir))

        fp_gen = os.path.join(out_dir, 'model_gen.pkl')
        fp_disc = os.path.join(out_dir, 'model_disc.pkl')

        params = cls._load_saved_params(out_dir)

        return cls(fp_gen, fp_disc, **params)

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        mps = super().model_params
        mps['optimizer_s'] = self.get_optimizer_config(self.optimizer_s)

        return mps

    @property
    def optimizer_s(self):
        """Get the tensorflow optimizer to perform gradient descent
        calculations for the spatial disc model.

        Returns
        -------
        tf.keras.optimizers.Optimizer
        """
        return self._optimizer_s

    def update_optimizer(self, option='generator', **kwargs):
        """Update optimizer by changing current configuration

        Parameters
        ----------
        option : str
            Which optimizer to update. Can be "generator", "temporal",
            "spatial", or "all".
        kwargs : dict
            kwargs to use for optimizer configuration update
        """

        if 'gen' in option.lower() or 'all' in option.lower():
            conf = self.get_optimizer_config(self.optimizer)
            conf.update(**kwargs)
            OptimizerClass = getattr(optimizers, conf['name'])
            self._optimizer = OptimizerClass.from_config(conf)

        if 'spatial' in option.lower() or 'all' in option.lower():
            conf = self.get_optimizer_config(self.optimizer_s)
            conf.update(**kwargs)
            OptimizerClass = getattr(optimizers, conf['name'])
            self._optimizer_s = OptimizerClass.from_config(conf)

    @property
    def disc(self):
        """Get the discriminator model.

        Returns
        -------
        phygnn.base.CustomNetwork
        """
        return self._disc

    @property
    def disc_weights(self):
        """Get a list of layer weights and bias terms for the discriminator
        model.

        Returns
        -------
        list
        """
        return self.disc.weights

    def discriminate(self, hi_res, norm_in=False):
        """Run the discriminator model on a hi resolution input field.

        Parameters
        ----------
        hi_res : tf.Tensor
            Real or fake high res data in a 4D tensor
            (n_obs, spatial_1, spatial_2, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The disc should always
            received normalized data with mean=0 stdev=1.

        Returns
        -------
        out : tf.Tensor
            Discriminator output logits
        """
        if norm_in and self._means is not None:
            hi_res = hi_res if isinstance(hi_res, tf.Tensor) else hi_res.copy()
            for i, (m, s) in enumerate(zip(self._means, self._stdevs)):
                islice = tuple([slice(None)] * (len(hi_res.shape) - 1) + [i])
                hi_res[islice] = (hi_res[islice] - m) / s

        out = self.disc.layers[0](hi_res)
        for i, layer in enumerate(self.disc.layers[1:]):
            try:
                out = layer(out)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, out.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        return out

    @property
    def weights(self):
        """Get a list of all the layer weights and bias terms for the
        generator, spatial discriminator, and temporal discriminator.
        """
        return self.generator_weights + self.disc_weights

    def calc_loss(self, hi_res_true, hi_res_gen,
                  weight_gen_advers=0.001, train_gen=True,
                  train_disc=False):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor
            Ground truth high resolution spatial data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        train_gen : bool
            True if generator is being trained, then loss=loss_gen
        train_disc : bool
            True if discriminator is being trained, then loss=loss_disc

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the network being trained
            (either generator or discriminator)
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        disc_out_true = self.discriminate(hi_res_true)
        disc_out_gen = self.discriminate(hi_res_gen)

        loss_gen_content = self.calc_loss_gen_content(
            hi_res_true, hi_res_gen)
        loss_gen_advers = self.calc_loss_gen_advers(disc_out_gen)
        loss_gen = loss_gen_content + weight_gen_advers * loss_gen_advers
        loss_disc = self.calc_loss_disc(disc_out_true, disc_out_gen)

        loss = None
        if train_gen:
            loss = loss_gen
        elif train_disc:
            loss = loss_disc

        loss_details = {'loss_gen': loss_gen,
                        'loss_disc': loss_disc,
                        'loss_gen_content': loss_gen_content,
                        'loss_gen_advers': loss_gen_advers,
                        }

        return loss, loss_details

    def calc_val_loss(self, batch_handler, weight_gen_advers, loss_details):
        """Calculate the validation loss at the current state of model training

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.SpatialBatchHandler
            SpatialBatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
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
            high_res_gen = self.generate(val_batch.low_res)
            _, v_loss_details = self.calc_loss(
                val_batch.high_res, high_res_gen,
                weight_gen_advers=weight_gen_advers,
                train_gen=False, train_disc=False)

            loss_details = self.update_loss_details(loss_details,
                                                    v_loss_details,
                                                    len(val_batch),
                                                    prefix='val_')

        return loss_details

    def train_epoch(self, batch_handler, weight_gen_advers, train_gen,
                    train_disc, disc_loss_bounds):
        """Train the GAN for one epoch.

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.SpatialBatchHandler
            SpatialBatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
        disc_loss_bounds : tuple
            Lower and upper bounds for the discriminator loss outside of which
            the discriminator will not train unless train_disc=True and
            train_gen=False.

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components
        """
        disc_th_low = np.min(disc_loss_bounds)
        disc_th_high = np.max(disc_loss_bounds)
        loss_details = {'n_obs': 0, 'train_loss_disc': 0}
        only_gen = train_gen and not train_disc
        only_disc = train_disc and not train_gen

        for ib, batch in enumerate(batch_handler):
            trained_gen = False
            trained_disc = False
            b_loss_details = {}
            loss_disc = loss_details['train_loss_disc']
            disc_too_good = loss_disc < disc_th_low
            gen_too_good = loss_disc > disc_th_high

            if only_gen or (train_gen and not gen_too_good):
                trained_gen = True
                b_loss_details = self.run_gradient_descent(
                    batch.low_res, batch.high_res, self.generator_weights,
                    weight_gen_advers=weight_gen_advers,
                    optimizer=self.optimizer,
                    train_gen=True, train_disc=False)

            if only_disc or (train_disc and not disc_too_good):
                trained_disc = True
                b_loss_details = self.run_gradient_descent(
                    batch.low_res, batch.high_res, self.disc_weights,
                    weight_gen_advers=weight_gen_advers,
                    optimizer=self.optimizer_s,
                    train_gen=False, train_disc=True)

            b_loss_details['gen_trained_frac'] = float(trained_gen)
            b_loss_details['disc_trained_frac'] = float(trained_disc)
            loss_details = self.update_loss_details(loss_details,
                                                    b_loss_details,
                                                    len(batch),
                                                    prefix='train_')

            logger.debug('Batch {} out of {} has epoch-average '
                         '(gen / disc) loss of: ({:.2e} / {:.2e}). '
                         'Trained (gen / disc): ({} / {})'
                         .format(ib, len(batch_handler),
                                 loss_details['train_loss_gen'],
                                 loss_details['train_loss_disc'],
                                 trained_gen, trained_disc))

        return loss_details

    def train(self, batch_handler, n_epoch, weight_gen_advers=0.001,
              train_gen=True, train_disc=True, disc_loss_bounds=(0.45, 0.6),
              checkpoint_int=None, out_dir='./spatial_gan_{epoch}',
              early_stop_on=None, early_stop_threshold=0.005,
              early_stop_n_epoch=5):
        """Train the GAN model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.SpatialBatchHandler
            SpatialBatchHandler object to iterate through
        n_epoch : int
            Number of epochs to train on
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
        disc_loss_bounds : tuple
            Lower and upper bounds for the discriminator loss outside of which
            the discriminator will not train unless train_disc=True and
            train_gen=False.
        checkpoint_int : int | None
            Epoch interval at which to save checkpoint models.
        out_dir : str
            Directory to save checkpoint GAN models. Should have {epoch} in
            the directory name. This directory will be created if it does not
            already exist.
        early_stop_on : str | None
            If not None, this should be a column in the training history to
            evaluate for early stopping (e.g. validation_loss_gen,
            validation_loss_disc). If this value in this history decreases by
            an absolute fractional relative difference of less than 0.01 for
            more than 5 epochs in a row, the training will stop early.
        early_stop_threshold : float
            The absolute relative fractional difference in validation loss
            between subsequent epochs below which an early termination is
            warranted. E.g. if val losses were 0.1 and 0.0998 the relative
            diff would be calculated as 0.0002 / 0.1 = 0.002 which would be
            less than the default thresold of 0.01 and would satisfy the
            condition for early termination.
        early_stop_n_epoch : int
            The number of consecutive epochs that satisfy the threshold that
            warrants an early stop.
        """

        self.set_norm_stats(batch_handler)
        self.set_feature_names(batch_handler)

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(
                columns=['elapsed_time'])
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        t0 = time.time()
        logger.info('Training model with weight_gen_advers: {} '
                    'for {} epochs starting at epoch {}'
                    .format(weight_gen_advers, n_epoch, epochs[0]))

        for epoch in epochs:
            loss_details = self.train_epoch(batch_handler, weight_gen_advers,
                                            train_gen, train_disc,
                                            disc_loss_bounds)

            loss_details = self.calc_val_loss(batch_handler, weight_gen_advers,
                                              loss_details)

            logger.info('Epoch {} of {} '
                        'generator train/val loss: {:.2e}/{:.2e} '
                        'discriminator train/val loss: {:.2e}/{:.2e}'
                        .format(epoch, epochs[-1],
                                loss_details['train_loss_gen'],
                                loss_details['val_loss_gen'],
                                loss_details['train_loss_disc'],
                                loss_details['val_loss_disc']))

            lr_g = self.get_optimizer_config(self.optimizer)['learning_rate']
            lr_s = self.get_optimizer_config(self.optimizer_s)['learning_rate']
            extras = {'weight_gen_advers': weight_gen_advers,
                      'disc_loss_bound_0': disc_loss_bounds[0],
                      'disc_loss_bound_1': disc_loss_bounds[1],
                      'learning_rate_gen': lr_g,
                      'learning_rate_s': lr_s}
            stop = self.finish_epoch(epoch, epochs, t0, loss_details,
                                     checkpoint_int, out_dir,
                                     early_stop_on, early_stop_threshold,
                                     early_stop_n_epoch, extras=extras)
            if stop:
                break
