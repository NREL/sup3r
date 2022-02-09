# -*- coding: utf-8 -*-
"""Sup3r model software"""
from abc import ABC, abstractmethod
import os
import time
import json
import logging
import numpy as np
import pprint
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import mean_squared_error
from rex.utilities.utilities import safe_json_load
from phygnn import CustomNetwork


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base sup3r GAN model."""

    def __init__(self, optimizer=None, learning_rate=1e-4,
                 history=None, version_record=None,
                 means=None, stdevs=None, name=None):
        """
        Parameters
        ----------
        optimizer : tensorflow.keras.optimizers | dict | None | str
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        history : pd.DataFrame | str | None
            Model training history with "epoch" index, str pointing to a saved
            history csv file with "epoch" as first column, or None for clean
            history
        version_record : dict | None
            Optional record of import package versions. None (default) will
            save active environment versions. A dictionary will be interpreted
            as versions from a loaded model and will be saved as an attribute.
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
        self.name = name
        self._gen = None
        self._learning_rate = learning_rate

        self._means = means
        self._stdevs = stdevs

        self._version_record = CustomNetwork._parse_versions(version_record)

        self._history = history
        if isinstance(self._history, str):
            self._history = pd.read_csv(self._history, index_col=0)

        self._optimizer = optimizer
        if isinstance(optimizer, dict):
            class_name = optimizer['name']
            OptimizerClass = getattr(optimizers, class_name)
            self._optimizer = OptimizerClass.from_config(optimizer)
        elif optimizer is None:
            self._optimizer = optimizers.Adam(learning_rate=learning_rate)

    @staticmethod
    def load_network(model, name):
        """Load a CustomNetwork object from hidden layers config, .json file
        config, or .pkl file saved pre=trained model.

        Parameters
        ----------
        model : str | dict
            Model hidden layers config, a .json with "hidden_layers" key, or a
            .pkl for a saved pre-trained model.
        name : str
            Name of the model to be loaded

        Returns
        -------
        model : phygnn.CustomNetwork
            CustomNetwork object initialized from the model input.
        """

        if isinstance(model, str) and model.endswith('.json'):
            model = safe_json_load(model)
            if 'hidden_layers' in model:
                model = model['hidden_layers']
            else:
                msg = ('Could not load model from json config, need '
                       '"hidden_layers" key at top level but only found: {}'
                       .format(model.keys()))
                logger.error(msg)
                raise KeyError(msg)

        elif isinstance(model, str) and model.endswith('.pkl'):
            model = CustomNetwork.load(model)

        if isinstance(model, list):
            model = CustomNetwork(hidden_layers=model, name=name)

        if not isinstance(model, CustomNetwork):
            msg = ('Something went wrong. Tried to load a custom network '
                   'but ended up with a model of type "{}"'
                   .format(type(model)))
            logger.error(msg)
            raise TypeError(msg)

        return model

    def set_norm_stats(self, batch_handler):
        """Set the normalization statistics associated with a data batch
        handler to model attributes.

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            Data batch handler with a multi_data_handler attribute with
            normalization stats.
        """

        if self._means is not None:
            logger.info('Setting new normalization statistics...')
            logger.info("Model's previous data mean values: {}"
                        .format(self._means))
            logger.info("Model's previous data stdev values: {}"
                        .format(self._stdevs))

        self._means = batch_handler.multi_data_handler.means
        self._stdevs = batch_handler.multi_data_handler.stds

        logger.info('Set data normalization mean values: {}'
                    .format(self._means))
        logger.info('Set data normalization stdev values: {}'
                    .format(self._stdevs))

    @property
    def generator(self):
        """Get the generative model.

        Returns
        -------
        phygnn.base.CustomNetwork
        """
        return self._gen

    @property
    def generator_weights(self):
        """Get a list of layer weights and bias terms for the generator model.

        Returns
        -------
        list
        """
        return self.generator.weights

    def generate(self, low_res, to_numpy=True, training=False):
        """Use the generator model to generate high res data from los res input

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data
        to_numpy : bool
            Flag to convert output from tensor to numpy array
        training : bool
            Flag for predict() used in the training routine. This is used
            to freeze the BatchNormalization and Dropout layers.

        Returns
        -------
        hi_res : np.ndarray
            Synthetically generated high-resolution data
        """
        return self.generator.predict(low_res, to_numpy=to_numpy,
                                      training=training)

    @property
    def version_record(self):
        """A record of important versions that this model was built with.

        Returns
        -------
        dict
        """
        return self._version_record

    @property
    def optimizer(self):
        """Get the tensorflow optimizer to perform gradient descent
        calculations.

        Returns
        -------
        tf.keras.optimizers.Optimizer
        """
        return self._optimizer

    @property
    def optimizer_config(self):
        """Get a config that defines the current GAN optimizer

        Returns
        -------
        dict
        """
        conf = self._optimizer.get_config()
        for k, v in conf.items():
            # need to convert numpy dtypes to float/int for json.dump()
            if np.issubdtype(type(v), np.floating):
                conf[k] = float(v)
            elif np.issubdtype(type(v), np.integer):
                conf[k] = int(v)
        return conf

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        means = (self._means if self._means is None
                 else [float(m) for m in self._means])
        stdevs = (self._stdevs if self._stdevs is None
                  else [float(s) for s in self._stdevs])

        model_params = {'name': self.name,
                        'version_record': self.version_record,
                        'learning_rate': self._learning_rate,
                        'optimizer': self.optimizer_config,
                        'means': means,
                        'stdevs': stdevs,
                        }

        return model_params

    @property
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        return self._history

    @property
    def weights(self):
        """Get a list of all the layer weights and bias terms for the
        generator, spatial discriminator, and temporal discriminator.
        """
        return self.generator_weights

    @staticmethod
    def early_stop(history, column, threshold=0.01, n_epoch=5):
        """Determine whether to stop training early based on nearly no change
        to validation loss for a certain number of consecutive epochs.

        Parameters
        ----------
        history : pd.DataFrame | None
            Model training history
        column : str
            Column from the model training history to evaluate for early
            termination.
        threshold : float
            The absolute relative fractional difference in validation loss
            between subsequent epochs below which an early termination is
            warranted. E.g. if val losses were 0.1 and 0.0998 the relative
            diff would be calculated as 0.0002 / 0.1 = 0.002 which would be
            less than the default thresold of 0.01 and would satisfy the
            condition for early termination.
        n_epoch : int
            The number of consecutive epochs that satisfy the threshold that
            warrants an early stop.

        Returns
        -------
        stop : bool
            Flag to stop training (True) or keep going (False).
        """
        stop = False

        if history is not None and len(history) > n_epoch + 1:
            diffs = np.abs(np.diff(history[column]))
            if all(diffs[-n_epoch:] < threshold):
                stop = True
                logger.info('Found early stop condition, loss values "{}" '
                            'have absolute relative differences less than '
                            'threshold {}: {}'
                            .format(column, threshold, diffs[-n_epoch:]))

        return stop

    def run_gradient_descent(self, low_res, hi_res_true, training_weights,
                             **calc_loss_kwargs):
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
        calc_loss_kwargs : dict
            Kwargs to pass to the self.calc_loss() method

        Returns
        -------
        loss_diagnostics : dict
            Namespace of the breakdown of loss components
        """

        with tf.GradientTape() as tape:
            tape.watch(training_weights)

            hi_res_gen = self.generate(low_res, to_numpy=False, training=True)
            loss_out = self.calc_loss(hi_res_true, hi_res_gen,
                                      **calc_loss_kwargs)
            loss, loss_diagnostics = loss_out

            grad = tape.gradient(loss, training_weights)

        self._optimizer.apply_gradients(zip(grad, training_weights))

        return loss_diagnostics

    @staticmethod
    def calc_loss_gen_content(hi_res_true, hi_res_gen):
        """Calculate the content loss term for the generator model.

        Parameters
        ----------
        hi_res_true : tf.Tensor | np.ndarray
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.

        Returns
        -------
        loss_gen_s : tf.Tensor
            0D tensor generator model loss for the content loss comparing the
            hi res ground truth to the hi res synthetically generated output.
        """

        loss_gen_content = mean_squared_error(hi_res_true, hi_res_gen)
        loss_gen_content = tf.reduce_mean(loss_gen_content)

        return loss_gen_content

    @staticmethod
    def calc_loss_gen_advers(disc_out_gen):
        """Calculate the adversarial component of the loss term for the
        generator model.

        Parameters
        ----------
        disc_out_gen : tf.Tensor
            Raw discriminator outputs from the discriminator model
            predicting only on hi_res_gen (not on hi_res_true).

        Returns
        -------
        loss_gen_advers : tf.Tensor
            0D tensor generator model loss for the adversarial component of the
            generator loss term.
        """

        # note that these have flipped labels from the discriminator
        # loss because of the opposite optimization goal
        loss_gen_advers = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_out_gen, labels=tf.ones_like(disc_out_gen))
        loss_gen_advers = tf.reduce_mean(loss_gen_advers)

        return loss_gen_advers

    @staticmethod
    def calc_loss_disc(disc_out_true, disc_out_gen):
        """Calculate the loss term for the discriminator model (either the
        spatial or temporal discriminator).

        Parameters
        ----------
        disc_out_true : tf.Tensor
            Raw discriminator outputs from the discriminator model predicting
            only on ground truth data hi_res_true (not on hi_res_gen).
        disc_out_gen : tf.Tensor
            Raw discriminator outputs from the discriminator model predicting
            only on synthetic data hi_res_gen (not on hi_res_true).

        Returns
        -------
        loss_disc : tf.Tensor
            0D tensor discriminator model loss for either the spatial or
            temporal component of the super resolution generated output.
        """

        # note that these have flipped labels from the generator
        # loss because of the opposite optimization goal
        logits = tf.concat([disc_out_true, disc_out_gen], axis=0)
        labels = tf.concat([tf.ones_like(disc_out_true),
                            tf.zeros_like(disc_out_gen)], axis=0)

        loss_disc = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                            labels=labels)
        loss_disc = tf.reduce_mean(loss_disc)

        return loss_disc

    @abstractmethod
    def calc_loss(self, hi_res_true, hi_res_gen):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor | np.ndarray
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the full GAN loss term.
        loss_diagnostics : dict
            Namespace of the breakdown of loss components
        """


class SpatialGan(BaseModel):
    """Spatial super resolution GAN model"""

    def __init__(self, gen_layers, disc_layers,
                 optimizer=None, learning_rate=1e-4,
                 history=None, version_record=None,
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
        history : pd.DataFrame | str | None
            Model training history with "epoch" index, str pointing to a saved
            history csv file with "epoch" as first column, or None for clean
            history
        version_record : dict | None
            Optional record of import package versions. None (default) will
            save active environment versions. A dictionary will be interpreted
            as versions from a loaded model and will be saved as an attribute.
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

        super().__init__(optimizer=optimizer, learning_rate=learning_rate,
                         history=history, version_record=version_record,
                         means=means, stdevs=stdevs, name=name)

        self._gen = self.load_network(gen_layers, 'Generator')
        self._disc = self.load_network(disc_layers, 'Discriminator')

    def save(self, out_dir):
        """Save the GAN with its sub-networks to a directory.

        Parameters
        ----------
        out_dir : str
            Directory to save GAN model files. This directory will be created
            if it does not already exist.
        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fp_gen = os.path.join(out_dir, 'model_gen.pkl')
        fp_disc = os.path.join(out_dir, 'model_disc.pkl')
        self.generator.save(fp_gen)
        self.disc.save(fp_disc)

        fp_history = None
        if isinstance(self.history, pd.DataFrame):
            fp_history = os.path.join(out_dir, 'history.csv')
            self.history.to_csv(fp_history)

        fp_params = os.path.join(out_dir, 'model_params.json')
        with open(fp_params, 'w') as f:
            params = self.model_params
            params['history'] = fp_history
            json.dump(params, f, sort_keys=True, indent=2)

        logger.info('Saved GAN to disk in directory: {}'.format(out_dir))

    @classmethod
    def load(cls, out_dir):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        out_dir : str
            Directory to load GAN model files from.
        """

        fp_gen = os.path.join(out_dir, 'model_gen.pkl')
        fp_disc = os.path.join(out_dir, 'model_disc.pkl')
        fp_params = os.path.join(out_dir, 'model_params.json')
        with open(fp_params, 'r') as f:
            params = json.load(f)

        if 'version_record' in params:
            logger.info('Loading GAN from disk that was created with the '
                        'following package versions: \n{}'
                        .format(pprint.pformat(params['version_record'],
                                               indent=4)))
            active_versions = CustomNetwork._parse_versions(None)
            logger.info('Active python environment versions: \n{}'
                        .format(pprint.pformat(active_versions, indent=4)))

        return cls(fp_gen, fp_disc, **params)

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

    def discriminate(self, hi_res, to_numpy=True, training=False):
        """Use the generator model to generate high res data from los res input

        Parameters
        ----------
        hi_res : np.ndarray | tf.Tensor
            Real or fake high res data in a 4D array or tensor
            (n_obs, spatial_1, spatial_2, n_features)
        to_numpy : bool
            Flag to convert output from tensor to numpy array
        training : bool
            Flag for predict() used in the training routine. This is used
            to freeze the BatchNormalization and Dropout layers.

        Returns
        -------
        hi_res : np.ndarray
            Synthetically generated high-resolution data
        """
        return self.disc.predict(hi_res, to_numpy=to_numpy,
                                 training=training)

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
        hi_res_true : tf.Tensor | np.ndarray
            Ground truth high resolution spatial data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the full GAN loss term. This can be a
            weighted summation of two individual loss terms from the
            generative + discriminative models.
        loss_diagnostics : dict
            Namespace of the breakdown of loss components
        """

        disc_out_true = self.discriminate(hi_res_true, to_numpy=False,
                                          training=True)
        disc_out_gen = self.discriminate(hi_res_gen, to_numpy=False,
                                         training=True)

        loss_gen_content = self.calc_loss_gen_content(hi_res_true, hi_res_gen)
        loss_gen_advers = self.calc_loss_gen_advers(disc_out_gen)
        loss_gen = loss_gen_content + weight_gen_advers * loss_gen_advers
        loss_disc = self.calc_loss_disc(disc_out_true, disc_out_gen)

        loss = None
        if train_gen:
            loss = loss_gen
        elif train_disc:
            loss = loss_disc

        loss_diagnostics = {'loss_gen': loss_gen,
                            'loss_disc': loss_disc,
                            }

        return loss, loss_diagnostics

    def train_generator(self, low_res, high_res, weight_gen_advers):
        """Train the generator network.

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data in a 4D array:
            (n_observations, spatial_1, spatial_2, features)
        hi_res_true : np.ndarray
            Real high-resolution data in a 4D array:
            (n_observations, spatial_1, spatial_2, features)
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
            Flag whether to train the discriminator for this set of epochs

        Returns
        -------
        loss_gen : tf.Tensor
            0D loss value for the generator training loss.
        loss_disc
            0D loss value for the discriminator training loss.
        """
        logger.debug('Training generator...')
        diag = self.run_gradient_descent(low_res, high_res,
                                         self.generator_weights,
                                         weight_gen_advers=weight_gen_advers,
                                         train_gen=True, train_disc=False)
        loss_gen = diag['loss_gen']
        loss_disc = diag['loss_disc']
        return loss_gen, loss_disc

    def train_disc(self, low_res, high_res, weight_gen_advers):
        """Train the discriminator network.

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data in a 4D array:
            (n_observations, spatial_1, spatial_2, features)
        hi_res_true : np.ndarray
            Real high-resolution data in a 4D array:
            (n_observations, spatial_1, spatial_2, features)
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
            Flag whether to train the discriminator for this set of epochs

        Returns
        -------
        loss_gen : tf.Tensor
            0D loss value for the generator training loss.
        loss_disc
            0D loss value for the discriminator training loss.
        """
        logger.debug('Training discriminator...')
        diag = self.run_gradient_descent(low_res, high_res, self.disc_weights,
                                         weight_gen_advers=weight_gen_advers,
                                         train_gen=False, train_disc=True)
        loss_gen = diag['loss_gen']
        loss_disc = diag['loss_disc']
        return loss_gen, loss_disc

    def train(self, batch_handler, n_epoch, weight_gen_advers=0.001,
              train_gen=True, train_disc=True, checkpoint_int=None,
              out_dir='./spatial_gan_{epoch}', early_stop_on=None):
        """Train the GAN model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.SpatialBatchHandler
            SpatialBatchHandler object to iterate through
        n_epoch : int
            Number of epochs to train on
        weight_gen_content : float
            Weight factor for the generative content loss term in the loss
            function.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
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
        """

        self.set_norm_stats(batch_handler)

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(
                columns=['elapsed_time'])
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        loss_gen = 0.0
        loss_disc = 0.0
        t0 = time.time()
        logger.info('Starting model training with {} epochs.'.format(n_epoch))
        for epoch in epochs:
            for ib, batch in enumerate(batch_handler):

                if train_gen and (not train_disc or loss_disc < 0.6):
                    loss_gen, loss_disc = self.train_generator(
                        batch.low_res, batch.high_res, weight_gen_advers)

                if train_disc and loss_disc > 0.45:
                    loss_gen, loss_disc = self.train_disc(
                        batch.low_res, batch.high_res, weight_gen_advers)

                logger.debug('Batch {} out of {} train gen/disc loss: '
                             '{:.2e}/{:.2e}'
                             .format(ib + 1, len(batch_handler),
                                     loss_gen, loss_disc))

            val_loss_gen = 0.0
            val_loss_disc = 0.0
            # # pylint: disable=C0200
            for iv in range(len(batch_handler.val_data.low_res)):
                low_res = batch_handler.val_data.low_res[iv:iv + 1]
                high_res = batch_handler.val_data.high_res[iv:iv + 1]
                high_res_gen = self.generate(low_res, to_numpy=False)
                _, diag = self.calc_loss(high_res, high_res_gen,
                                         weight_gen_advers=weight_gen_advers)
                val_loss_gen += diag['loss_gen'].numpy()
                val_loss_disc += diag['loss_disc'].numpy()

            val_loss_gen /= len(batch_handler.val_data.low_res)
            val_loss_disc /= len(batch_handler.val_data.low_res)
            loss_gen = loss_gen.numpy()
            loss_disc = loss_disc.numpy()

            logger.info('Epoch {} of {} generator train/val loss: '
                        '{:.2e}/{:.2e} '
                        'discriminator train/val loss: {:.2e}/{:.2e}'
                        .format(epoch + 1, len(epochs), loss_gen, val_loss_gen,
                                loss_disc, val_loss_disc))

            self._history.at[epoch, 'elapsed_time'] = time.time() - t0
            self._history.at[epoch, 'training_loss_gen'] = loss_gen
            self._history.at[epoch, 'training_loss_disc'] = loss_disc
            self._history.at[epoch, 'validation_loss_gen'] = val_loss_gen
            self._history.at[epoch, 'validation_loss_disc'] = val_loss_disc

            if checkpoint_int is not None and (epoch % checkpoint_int) == 0:
                msg = ('GAN output dir for checkpoint models should have '
                       f'{"{epoch}"} but did not: {out_dir}')
                assert '{epoch}' in out_dir, msg
                self.save(out_dir.format(epoch=epoch))

            if early_stop_on is not None and early_stop_on in self._history:
                stop = self.early_stop(self._history, early_stop_on)
                if stop:
                    self.save(out_dir.format(epoch=epoch))
                    break


class SpatioTemporalGan(BaseModel):
    """Spatio temporal super resolution GAN model."""

    def __init__(self, gen_layers, disc_t_layers, disc_s_layers,
                 **kwargs):
        """
        Parameters
        ----------
        gen_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            generative super resolving model. Can also be a str filepath to a
            .json config file containing the input layers argument or a .pkl
            for a saved pre-trained model.
        disc_t_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative temporal model. Can also be a str filepath to a
            .json config file containing the input layers argument or a .pkl
            for a saved pre-trained model.
        disc_s_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative spatial model. Can also be a str filepath to a
            .json config file containing the input layers argument or a .pkl
            for a saved pre-trained model.
        kwargs : dict
            Key word args for BaseModel init.
        """

        super().__init__(**kwargs)

        self._gen = self.load_network(gen_layers, 'Generator')
        self._disc_t = self.load_network(disc_t_layers, 'TemporalDisc')
        self._disc_s = self.load_network(disc_s_layers, 'SpatialDisc')

    @property
    def disc_spatial(self):
        """Get the spatial discriminator model.

        Returns
        -------
        phygnn.base.CustomNetwork
        """
        return self._disc_s

    @property
    def disc_spatial_weights(self):
        """Get a list of layer weights and bias terms for the spatial
        discriminator model.

        Returns
        -------
        list
        """
        return self.disc_spatial.weights

    @property
    def disc_temporal(self):
        """Get the temporal discriminator model.

        Returns
        -------
        phygnn.base.CustomNetwork
        """
        return self._disc_t

    @property
    def disc_temporal_weights(self):
        """Get a list of layer weights and bias terms for the temporal
        discriminator model.

        Returns
        -------
        list
        """
        return self.disc_temporal.weights

    @property
    def weights(self):
        """Get a list of all the layer weights and bias terms for the
        generator, spatial discriminator, and temporal discriminator.
        """
        return (self.generator_weights + self.disc_spatial_weights
                + self.disc_temporal_weights)

    def calc_loss(self, hi_res_true, hi_res_gen,
                  weight_gen_advers_s=0.001, weight_gen_advers_t=0.001,
                  train_gen=True, train_disc_s=False, train_disc_t=False):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor | np.ndarray
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.
        weight_gen_advers_s : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        weight_gen_advers_t : float
            Weight factor for the adversarial loss component of the generator
            vs. the temporal discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc_s : bool
            Flag whether to train the spatial discriminator for this set of
            epochs
        train_disc_t : bool
            Flag whether to train the temporal discriminator for this set of
            epochs

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the full GAN loss term. This can be a
            weighted summation of up to four individual loss terms from the
            generative / discriminative models and their respective spatial /
            temporal components or models.
        loss_diagnostics : dict
            Namespace of the breakdown of loss components
        """

        disc_out_spatial_true = self.disc_spatial.predict(hi_res_true)
        disc_out_spatial_gen = self.disc_spatial.predict(hi_res_gen)

        disc_out_temporal_true = self.disc_temporal.predict(hi_res_true)
        disc_out_temporal_gen = self.disc_temporal.predict(hi_res_gen)

        loss_gen_content = self.calc_loss_gen_content(hi_res_true, hi_res_gen)
        loss_gen_advers_s = self.calc_loss_gen_advers(disc_out_spatial_gen)
        loss_gen_advers_t = self.calc_loss_gen_advers(disc_out_temporal_gen)
        loss_gen = (loss_gen_content
                    + weight_gen_advers_s * loss_gen_advers_s
                    + weight_gen_advers_t * loss_gen_advers_t)

        loss_disc_s = self.calc_loss_disc(disc_out_spatial_true,
                                          disc_out_spatial_gen)

        loss_disc_t = self.calc_loss_disc(disc_out_temporal_true,
                                          disc_out_temporal_gen)

        loss = None
        if train_gen:
            loss = loss_gen
        elif train_disc_s:
            loss = loss_disc_s
        elif train_disc_t:
            loss = loss_disc_t

        loss_diagnostics = {'loss_gen': loss_gen,
                            'loss_disc_s': loss_disc_s,
                            'loss_disc_t': loss_disc_t,
                            }

        return loss, loss_diagnostics
