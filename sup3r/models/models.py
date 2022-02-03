# -*- coding: utf-8 -*-
"""Sup3r model software"""
from abc import ABC, abstractmethod
import time
import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import mean_squared_error
from rex.utilities.utilities import safe_json_load
from phygnn import CustomNetwork


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base sup3r GAN model."""

    def __init__(self, optimizer=None, learning_rate=1e-4):
        """
        Parameters
        ----------
        optimizer : tensorflow.keras.optimizers | dict | None
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        """
        self.name = None
        self._history = None
        self._gen = None

        self._optimizer = optimizer
        if isinstance(optimizer, dict):
            class_name = optimizer['name']
            OptimizerClass = getattr(optimizers, class_name)
            self._optimizer = OptimizerClass.from_config(optimizer)
        elif optimizer is None:
            self._optimizer = optimizers.Adam(learning_rate=learning_rate)

    @staticmethod
    def load_network(model, name):
        """Load a CustomNetwork object from hidden layers config or json file.

        Parameters
        ----------
        model : str | dict
            Model hidden layers config or a str pointing to a json with
            "hidden_layers" key
        name : str
            Name of the model to be loaded

        Returns
        -------
        model : phygnn.CustomNetwork
            CustomNetwork object initialized from the model input.
        """

        if isinstance(model, str):
            model = safe_json_load(model)
            if 'hidden_layers' in model:
                model = model['hidden_layers']
            else:
                msg = ('Could not load model from json config, need '
                       '"hidden_layers" key at top level but only found: {}'
                       .format(model.keys()))
                logger.error(msg)
                raise KeyError(msg)

        if isinstance(model, list):
            model = CustomNetwork(hidden_layers=model, name=name)

        if not isinstance(model, CustomNetwork):
            msg = ('Something went wrong. Tried to load a custom network '
                   'but ended up with a model of type "{}"'
                   .format(type(model)))

        return model

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

    @property
    @abstractmethod
    def training_weights(self):
        """Get a list of layer weights that are to-be-trained based on the
        current loss weight values."""

    def run_gradient_descent(self, low_res, hi_res_true):
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

        with tf.GradientTape() as tape:
            tape.watch(self.training_weights)

            hi_res_gen = self.generate(low_res, to_numpy=False, training=True)
            loss_out = self.calc_loss(hi_res_true, hi_res_gen)
            loss, loss_diagnostics = loss_out

            grad = tape.gradient(loss, self.training_weights)

        self._optimizer.apply_gradients(zip(grad, self.training_weights))

        return loss, loss_diagnostics

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

    def __init__(self, gen_layers, disc_layers, weight_gen_content=1.0,
                 weight_gen_advers=1.0, weight_disc=1.0,
                 optimizer=None, learning_rate=1e-4):
        """
        Parameters
        ----------
        gen_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            generative super resolving model. Can also be a str filepath to a
            json config file containing the input layers argument.
        disc_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative model. Can also be a str filepath to a json
            config file containing the input layers argument.
        weight_gen_content : float
            Weight factor for the generative content loss term in the loss
            function.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        weight_disc : float
            Weight factor for the spatial discriminator loss term in the loss
            function.
        optimizer : tensorflow.keras.optimizers | dict | None
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        """

        super().__init__(optimizer=optimizer, learning_rate=learning_rate)

        self.weight_gen_content = weight_gen_content
        self.weight_gen_advers = weight_gen_advers
        self.weight_disc = weight_disc

        self._gen = self.load_network(gen_layers, 'Generator')
        self._disc = self.load_network(disc_layers, 'Discriminator')

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

    @property
    def weights(self):
        """Get a list of all the layer weights and bias terms for the
        generator, spatial discriminator, and temporal discriminator.
        """
        return self.generator_weights + self.disc_weights

    @property
    def training_weights(self):
        """Get a list of layer weights that are to-be-trained based on the
        current loss weight values."""

        training_weights = []
        if self.weight_gen_content > 0:
            training_weights += self.generator_weights

        if self.weight_disc > 0:
            training_weights += self.disc_weights

        return training_weights

    def calc_loss(self, hi_res_true, hi_res_gen):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor | np.ndarray
            Ground truth high resolution spatial data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the full GAN loss term. This can be a
            weighted summation of two individual loss terms from the
            generative + discriminative models.
        loss_diagnostics : dict
            Namespace of the breakdown of loss components
        """

        loss_gen_content = tf.constant(0.0, dtype=tf.float32)
        loss_gen_advers = tf.constant(0.0, dtype=tf.float32)
        loss_disc = tf.constant(0.0, dtype=tf.float32)

        disc_out_true = self.disc.predict(hi_res_true)
        disc_out_gen = self.disc.predict(hi_res_gen)

        if self.weight_gen_content > 0:
            loss_gen_content = self.calc_loss_gen_content(hi_res_true,
                                                          hi_res_gen)

        if self.weight_gen_advers > 0:
            loss_gen_advers = self.calc_loss_gen_advers(disc_out_gen)

        if self.weight_disc > 0:
            loss_disc = self.calc_loss_disc(disc_out_true, disc_out_gen)

        loss = (self.weight_gen_content * loss_gen_content
                + self.weight_gen_advers * loss_gen_advers
                + self.weight_disc * loss_disc)

        loss_diagnostics = {'loss_gen_content': loss_gen_content,
                            'loss_gen_advers': loss_gen_advers,
                            'loss_disc': loss_disc,
                            }

        return loss, loss_diagnostics

    def train(self, batch_handler, n_epoch):
        """Train the GAN model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.SpatialBatchHandler
            SpatialBatchHandler object to iterate through
        """
        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(
                columns=['elapsed_time',
                         'training_loss',
                         'validation_loss',
                         ])
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        tr_loss = None
        t0 = time.time()
        logger.info('Starting model training with {} epochs.'.format(n_epoch))
        for epoch in epochs:
            for i, batch in enumerate(batch_handler):
                tr_loss, _ = self.run_gradient_descent(batch.low_res,
                                                       batch.high_res)
                logger.debug('\t - Batch {} of {} has training loss of {:.2e}'
                             .format(i + 1, len(batch_handler.batch_indices),
                                     tr_loss))

            val_loss = tf.constant(0.0, dtype=tf.float32)
#            hi_res_val_gen = self.generate(batch_handler.val_data.low_res,
#                                           to_numpy=False)
#            val_loss, _ = self.calc_loss(batch_handler.val_data.high_res,
#                                         hi_res_val_gen)

            logger.info('Epoch {} train loss: {:.2e} '
                        'val loss: {:.2e} for "{}"'
                        .format(epoch, tr_loss, val_loss, self.name))

            self._history.at[epoch, 'elapsed_time'] = time.time() - t0
            self._history.at[epoch, 'training_loss'] = tr_loss.numpy()
            self._history.at[epoch, 'validation_loss'] = val_loss.numpy()


class SpatioTemporalGan(BaseModel):
    """Spatio temporal super resolution GAN model."""

    def __init__(self, gen_layers, disc_t_layers, disc_s_layers,
                 weight_gen_content=1.0, weight_gen_advers_s=1.0,
                 weight_gen_advers_t=1.0, weight_disc_s=1.0,
                 weight_disc_t=1.0, optimizer=None, learning_rate=1e-4):
        """
        Parameters
        ----------
        gen_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            generative super resolving model. Can also be a str filepath to a
            json config file containing the input layers argument.
        disc_t_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative temporal model. Can also be a str filepath to a json
            config file containing the input layers argument.
        disc_s_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative spatial model. Can also be a str filepath to a json
            config file containing the input layers argument.
        weight_gen_content : float
            Weight factor for the generative content loss term in the loss
            function.
        weight_gen_advers_s : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        weight_gen_advers_t : float
            Weight factor for the adversarial loss component of the generator
            vs. the temporal discriminator.
        weight_disc_s : float
            Weight factor for the spatial discriminator loss term in the loss
            function.
        weight_disc_t : float
            Weight factor for the temporal discriminator loss term in the loss
            function.
        optimizer : tensorflow.keras.optimizers | dict | None
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        """

        super().__init__(optimizer=optimizer, learning_rate=learning_rate)

        self.weight_gen_content = weight_gen_content
        self.weight_gen_advers_s = weight_gen_advers_s
        self.weight_gen_advers_t = weight_gen_advers_t
        self.weight_disc_t = weight_disc_t
        self.weight_disc_s = weight_disc_s

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

    @property
    def training_weights(self):
        """Get a list of layer weights that are to-be-trained based on the
        current loss weight values."""

        # do you even lift bro?
        training_weights = []
        if self.weight_gen_content > 0:
            training_weights += self.generator_weights

        if self.weight_disc_s > 0:
            training_weights += self.disc_spatial_weights

        if self.weight_disc_t > 0:
            training_weights += self.disc_temporal_weights

        return training_weights

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
            0D tensor representing the full GAN loss term. This can be a
            weighted summation of up to four individual loss terms from the
            generative / discriminative models and their respective spatial /
            temporal components or models.
        loss_diagnostics : dict
            Namespace of the breakdown of loss components
        """

        loss_gen_content = tf.constant(0.0, dtype=tf.float32)
        loss_gen_advers_s = tf.constant(0.0, dtype=tf.float32)
        loss_gen_advers_t = tf.constant(0.0, dtype=tf.float32)
        loss_disc_s = tf.constant(0.0, dtype=tf.float32)
        loss_disc_t = tf.constant(0.0, dtype=tf.float32)

        disc_out_spatial_true = self.disc_spatial.predict(hi_res_true)
        disc_out_spatial_gen = self.disc_spatial.predict(hi_res_gen)

        disc_out_temporal_true = self.disc_temporal.predict(hi_res_true)
        disc_out_temporal_gen = self.disc_temporal.predict(hi_res_gen)

        if self.weight_gen_content > 0:
            loss_gen_content = self.calc_loss_gen_content(hi_res_true,
                                                          hi_res_gen)

        if self.weight_gen_advers_s > 0:
            loss_gen_advers_s = self.calc_loss_gen_advers(disc_out_spatial_gen)

        if self.weight_gen_advers_t > 0:
            loss_gen_advers_t = self.calc_loss_gen_advers(
                disc_out_temporal_gen)

        if self.weight_disc_s > 0:
            loss_disc_s = self.calc_loss_disc(disc_out_spatial_true,
                                              disc_out_spatial_gen)

        if self.weight_disc_t > 0:
            loss_disc_t = self.calc_loss_disc(disc_out_temporal_true,
                                              disc_out_temporal_gen)

        loss = (self.weight_gen_content * loss_gen_content
                + self.weight_gen_advers_s * loss_gen_advers_s
                + self.weight_gen_advers_t * loss_gen_advers_t
                + self.weight_disc_s * loss_disc_s
                + self.weight_disc_t * loss_disc_t)

        loss_diagnostics = {'loss_gen_content': loss_gen_content,
                            'loss_gen_advers_s': loss_gen_advers_s,
                            'loss_gen_advers_t': loss_gen_advers_t,
                            'loss_disc_s': loss_disc_s,
                            'loss_disc_t': loss_disc_t,
                            }

        return loss, loss_diagnostics
