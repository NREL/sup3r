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
from warnings import warn
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import mean_squared_error
from rex.utilities.utilities import safe_json_load
from phygnn import CustomNetwork


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base sup3r GAN model."""

    def __init__(self, optimizer=None, learning_rate=1e-4,
                 history=None, version_record=None, meta=None,
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
        self.name = name
        self._gen = None
        self._meta = meta if meta is not None else {}

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

    def load_network(self, model, name):
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
            self._meta[f'config_{name}'] = model
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
        self.generator.save(fp_gen)

        fp_history = None
        if isinstance(self.history, pd.DataFrame):
            fp_history = os.path.join(out_dir, 'history.csv')
            self.history.to_csv(fp_history)

        fp_params = os.path.join(out_dir, 'model_params.json')
        with open(fp_params, 'w') as f:
            params = self.model_params
            json.dump(params, f, sort_keys=True, indent=2)

    @staticmethod
    def _load_saved_params(out_dir):
        """Load saved GAN model_params (you need this and the gen+disc models
        to load a full GAN).

        Parameters
        ----------
        out_dir : str
            Directory to load GAN model files from.

        Returns
        -------
        params : dict
            Model parameters loaded from disk json file. This should be the
            same as self.model_params with and additional 'history' entry.
            Should be all the kwargs you need to init a GAN model.
        """

        fp_params = os.path.join(out_dir, 'model_params.json')
        fp_history = os.path.join(out_dir, 'history.csv')

        with open(fp_params, 'r') as f:
            params = json.load(f)

        # using the saved model dir makes this more portable
        params['history'] = fp_history

        if 'version_record' in params:
            logger.info('Loading GAN from disk that was created with the '
                        'following package versions: \n{}'
                        .format(pprint.pformat(params['version_record'],
                                               indent=2)))
            active_versions = CustomNetwork._parse_versions(None)
            logger.info('Active python environment versions: \n{}'
                        .format(pprint.pformat(active_versions, indent=2)))

        return params

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

        self._means = batch_handler.means
        self._stdevs = batch_handler.stds

        logger.info('Set data normalization mean values: {}'
                    .format(self._means))
        logger.info('Set data normalization stdev values: {}'
                    .format(self._stdevs))

    def set_feature_names(self, batch_handler):
        """Set the list of feature names input/output to/from the generative
        model and the discriminator(s)"""

        if self.training_features is None:
            self.meta['training_features'] = batch_handler.training_features
        elif self.training_features != batch_handler.training_features:
            msg = ('GAN was previously trained on feature list {} but '
                   'received new features {}'
                   .format(self.training_features,
                           batch_handler.training_features))
            logger.error(msg)
            raise KeyError(msg)

        if self.output_features is None:
            self.meta['output_features'] = batch_handler.output_features
        elif self.output_features != batch_handler.output_features:
            msg = ('GAN was previously trained to output features {} but '
                   'received data to output new features {}'
                   .format(self.output_features,
                           batch_handler.output_features))
            logger.error(msg)
            raise KeyError(msg)

    @staticmethod
    def seed(s=0):
        """
        Set the random seed for reproducible results.

        Parameters
        ----------
        s : int
            Random seed
        """
        CustomNetwork.seed(s=s)

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

    @tf.function
    def generate(self, low_res, norm_in=False):
        """Use the generator model to generate high res data from los res input

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.

        Returns
        -------
        hi_res : tf.Tensor
            Synthetically generated high-resolution data
        """

        if norm_in and self._means is not None:
            if isinstance(low_res, tf.Tensor):
                msg = 'Cannot normalize tensor input, must be array'
                logger.error(msg)
                raise TypeError(msg)

            low_res = low_res.copy()
            for i, (m, s) in enumerate(zip(self._means, self._stdevs)):
                low_res[..., i] = (low_res[..., i] - m) / s

        hi_res = self.generator.layers[0](low_res)
        for layer in self.generator.layers[1:]:
            hi_res = layer(hi_res)

        return hi_res

    def un_norm_output(self, hi_res):
        """Un-normalize synthetically generated output data to physical units

        Parameters
        ----------
        hi_res : tf.Tensor | np.ndarray
            Synthetically generated high-resolution data

        Returns
        -------
        hi_res : np.ndarray
            Synthetically generated high-resolution data
        """
        if self._means is not None:
            if isinstance(hi_res, tf.Tensor):
                hi_res = hi_res.numpy()

            for i, feature in enumerate(self.training_features):
                if feature in self.output_features:
                    m = self._means[i]
                    s = self._stdevs[i]
                    j = self.output_features.index(feature)
                    hi_res[..., j] = (hi_res[..., j] * s) + m

        return hi_res

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

    def update_optimizer(self, **kwargs):
        """Update optimizer by changing current configuration

        Parameters
        kwargs : dict
            kwargs to use for optimizer configuration update

        """
        conf = self.optimizer_config
        conf.update(**kwargs)
        OptimizerClass = getattr(optimizers, conf['name'])
        self._optimizer = OptimizerClass.from_config(conf)

    @property
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""
        return self._meta

    @property
    def training_features(self):
        """Get the list of input feature names that the generative model was
        trained on."""
        return self.meta.get('training_features', None)

    @property
    def output_features(self):
        """Get the list of output feature names that the generative model
        outputs and that the discriminator predicts on."""
        return self.meta.get('output_features', None)

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
                        'optimizer': self.optimizer_config,
                        'means': means,
                        'stdevs': stdevs,
                        'meta': self.meta,
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
    def update_loss_details(loss_details, new_data, batch_len, prefix=None):
        """Update a dictionary of loss_details with loss information from a new
        batch.

        Parameters
        ----------
        loss_details : dict
            Namespace of the breakdown of loss components where each value is a
            running average at the current state in the epoch.
        new_data : dict
            Namespace of the breakdown of loss components for a single new
            batch.
        batch_len : int
            Length of the incomming batch.
        prefix : None | str
            Option to prefix the names of the loss data when saving to the
            loss_details dictionary.

        Returns
        -------
        loss_details : dict
            Same as input loss_details but with running averages updated.
        """
        assert 'n_obs' in loss_details, 'loss_details must have n_obs to start'
        prior_n_obs = loss_details['n_obs']
        new_n_obs = prior_n_obs + batch_len

        for key, new_value in new_data.items():
            key = key if prefix is None else prefix + key
            new_value = (new_value if not isinstance(new_value, tf.Tensor)
                         else new_value.numpy())

            if key in loss_details:
                saved_value = loss_details[key]
                saved_value *= prior_n_obs
                saved_value += batch_len * new_value
                saved_value /= new_n_obs
                loss_details[key] = saved_value
            else:
                loss_details[key] = new_value

        loss_details['n_obs'] = new_n_obs

        return loss_details

    @staticmethod
    def log_loss_details(loss_details, level='INFO'):
        """Log the loss details to the module logger.

        Parameters
        ----------
        loss_details : dict
            Namespace of the breakdown of loss components where each value is a
            running average at the current state in the epoch.
        level : str
            Log level (e.g. INFO, DEBUG)
        """
        for k, v in sorted(loss_details.items()):
            if k != 'n_obs':
                if level.lower() == 'info':
                    logger.info('\t{}: {:.2e}'.format(k, v))
                else:
                    logger.debug('\t{}: {:.2e}'.format(k, v))

    @staticmethod
    def early_stop(history, column, threshold=0.005, n_epoch=5):
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

    def finish_epoch(self, epoch, epochs, t0, loss_details,
                     checkpoint_int, out_dir,
                     early_stop_on, early_stop_threshold,
                     early_stop_n_epoch, extras=None):
        """Perform finishing checks after an epoch is done training

        Parameters
        ----------
        epoch : int
            Epoch number that is finishing
        epochs : list
            List of epochs being iterated through
        t0 : float
            Starting time of training.
        loss_details : dict
            Namespace of the breakdown of loss components
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
        extras : dict | None
            Extra kwargs/parameters to save in the epoch history.

        Returns
        -------
        stop : bool
            Flag to early stop training.
        """

        self.log_loss_details(loss_details)

        self._history.at[epoch, 'elapsed_time'] = time.time() - t0
        for key, value in loss_details.items():
            if key != 'n_obs':
                self._history.at[epoch, key] = value

        last_epoch = epoch == epochs[-1]
        chp = checkpoint_int is not None and (epoch % checkpoint_int) == 0
        if last_epoch or chp:
            msg = ('GAN output dir for checkpoint models should have '
                   f'{"{epoch}"} but did not: {out_dir}')
            assert '{epoch}' in out_dir, msg
            self.save(out_dir.format(epoch=epoch))

        stop = False
        if early_stop_on is not None and early_stop_on in self._history:
            stop = self.early_stop(self._history, early_stop_on,
                                   threshold=early_stop_threshold,
                                   n_epoch=early_stop_n_epoch)
            if stop:
                self.save(out_dir.format(epoch=epoch))

        if extras is not None:
            for k, v in extras.items():
                self._history.at[epoch, k] = v

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
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(training_weights)

            hi_res_gen = self.generate(low_res)
            loss_out = self.calc_loss(hi_res_true, hi_res_gen,
                                      **calc_loss_kwargs)
            loss, loss_details = loss_out

            grad = tape.gradient(loss, training_weights)

        self._optimizer.apply_gradients(zip(grad, training_weights))

        return loss_details

    @staticmethod
    @tf.function
    def calc_loss_gen_content(hi_res_true, hi_res_gen):
        """Calculate the content loss term for the generator model.

        Parameters
        ----------
        hi_res_true : tf.Tensor
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
    @tf.function
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
    @tf.function
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
    @tf.function
    def calc_loss(self, hi_res_true, hi_res_gen):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the full GAN loss term.
        loss_details : dict
            Namespace of the breakdown of loss components
        """


class SpatialGan(BaseModel):
    """Spatial super resolution GAN model"""

    def __init__(self, gen_layers, disc_layers,
                 optimizer=None, learning_rate=1e-4,
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

        super().__init__(optimizer=optimizer, learning_rate=learning_rate,
                         history=history, version_record=version_record,
                         meta=meta, means=means, stdevs=stdevs, name=name)

        self._gen = self.load_network(gen_layers, 'generator')
        self._disc = self.load_network(disc_layers, 'discriminator')

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
        for layer in self.disc.layers[1:]:
            out = layer(out)

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

        loss_gen_content = self.calc_loss_gen_content(hi_res_true, hi_res_gen)
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
                    train_gen=True, train_disc=False)

            if only_disc or (train_disc and not disc_too_good):
                trained_disc = True
                b_loss_details = self.run_gradient_descent(
                    batch.low_res, batch.high_res, self.disc_weights,
                    weight_gen_advers=weight_gen_advers,
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

            extras = {'weight_gen_advers': weight_gen_advers,
                      'disc_loss_bound_0': disc_loss_bounds[0],
                      'disc_loss_bound_1': disc_loss_bounds[1],
                      'learning_rate': self.optimizer_config['learning_rate']}
            stop = self.finish_epoch(epoch, epochs, t0, loss_details,
                                     checkpoint_int, out_dir,
                                     early_stop_on, early_stop_threshold,
                                     early_stop_n_epoch, extras=extras)
            if stop:
                break


class SpatioTemporalGan(BaseModel):
    """Spatio temporal super resolution GAN model."""

    def __init__(self, gen_layers, disc_s_layers, disc_t_layers,
                 optimizer=None, learning_rate=1e-4,
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
        disc_s_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative spatial model. Can also be a str filepath to a
            .json config file containing the input layers argument or a .pkl
            for a saved pre-trained model.
        disc_t_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative temporal model. Can also be a str filepath to a
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

        super().__init__(optimizer=optimizer, learning_rate=learning_rate,
                         history=history, version_record=version_record,
                         meta=meta, means=means, stdevs=stdevs, name=name)

        self._gen = self.load_network(gen_layers, 'generator')
        self._disc_s = self.load_network(disc_s_layers, 'spatial_disc')
        self._disc_t = self.load_network(disc_t_layers, 'temporal_disc')

    def save(self, out_dir):
        """Save the GAN with its sub-networks to a directory.

        Parameters
        ----------
        out_dir : str
            Directory to save GAN model files. This directory will be created
            if it does not already exist.
        """

        fp_disc_s = os.path.join(out_dir, 'model_disc_s.pkl')
        fp_disc_t = os.path.join(out_dir, 'model_disc_t.pkl')
        self.disc_spatial.save(fp_disc_s)
        self.disc_temporal.save(fp_disc_t)
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
        out : SpatioTemporalGan
            Returns a pretrained SpatioTemporalGan model that was previously
            saved to out_dir
        """

        logger.info('Loading GAN from disk in directory: {}'.format(out_dir))

        fp_gen = os.path.join(out_dir, 'model_gen.pkl')
        fp_disc_s = os.path.join(out_dir, 'model_disc_s.pkl')
        fp_disc_t = os.path.join(out_dir, 'model_disc_t.pkl')

        params = cls._load_saved_params(out_dir)

        return cls(fp_gen, fp_disc_s, fp_disc_t, **params)

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

    @tf.function
    def discriminate_s(self, hi_res, norm_in=False):
        """Run the spatial discriminator model on a hi resolution input field

        Parameters
        ----------
        hi_res : tf.Tensor
            Real or fake high res data in a 5D tensor
            (n_obs, spatial_1, spatial_2, temporal, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The disc should always
            received normalized data with mean=0 stdev=1.

        Returns
        -------
        out : tf.Tensor
            Spatial discriminator output logits
        """
        if norm_in and self._means is not None:
            hi_res = hi_res if isinstance(hi_res, tf.Tensor) else hi_res.copy()
            for i, (m, s) in enumerate(zip(self._means, self._stdevs)):
                islice = tuple([slice(None)] * (len(hi_res.shape) - 1) + [i])
                hi_res[islice] = (hi_res[islice] - m) / s

        out = self.disc_spatial.layers[0](hi_res)
        for layer in self.disc_spatial.layers[1:]:
            out = layer(out)

        return out

    @tf.function
    def discriminate_t(self, hi_res, norm_in=False):
        """Run the temporal discriminator model on a hi resolution input field

        Parameters
        ----------
        hi_res : tf.Tensor
            Real or fake high res data in a 5D tensor
            (n_obs, spatial_1, spatial_2, temporal, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The disc should always
            received normalized data with mean=0 stdev=1.

        Returns
        -------
        out : tf.Tensor
            Temporal discriminator output logits
        """
        if norm_in and self._means is not None:
            hi_res = hi_res if isinstance(hi_res, tf.Tensor) else hi_res.copy()
            for i, (m, s) in enumerate(zip(self._means, self._stdevs)):
                islice = tuple([slice(None)] * (len(hi_res.shape) - 1) + [i])
                hi_res[islice] = (hi_res[islice] - m) / s

        out = self.disc_temporal.layers[0](hi_res)
        for layer in self.disc_temporal.layers[1:]:
            out = layer(out)

        return out

    @property
    def weights(self):
        """Get a list of all the layer weights and bias terms for the
        generator, spatial discriminator, and temporal discriminator.
        """
        return (self.generator_weights + self.disc_spatial_weights
                + self.disc_temporal_weights)

    @tf.function
    def calc_loss(self, hi_res_true, hi_res_gen,
                  weight_gen_advers_s=0.001, weight_gen_advers_t=0.001,
                  train_gen=True, train_disc_s=False, train_disc_t=False):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor
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
            True if generator is being trained, then loss=loss_gen
        train_disc_s : bool
            True if spatial disc is being trained, then loss=loss_disc_s
        train_disc_t : bool
            True if temporal disc is being trained, then loss=loss_disc_t

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the network being trained
            (either generator or one of the discriminators)
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        disc_out_spat_true = self.discriminate_s(hi_res_true)
        disc_out_spat_gen = self.discriminate_s(hi_res_gen)

        disc_out_temp_true = self.discriminate_t(hi_res_true)
        disc_out_temp_gen = self.discriminate_t(hi_res_gen)

        loss_gen_content = self.calc_loss_gen_content(hi_res_true, hi_res_gen)
        loss_gen_advers_s = self.calc_loss_gen_advers(disc_out_spat_gen)
        loss_gen_advers_t = self.calc_loss_gen_advers(disc_out_temp_gen)
        loss_gen = (loss_gen_content
                    + weight_gen_advers_s * loss_gen_advers_s
                    + weight_gen_advers_t * loss_gen_advers_t)

        loss_disc_s = self.calc_loss_disc(disc_out_spat_true,
                                          disc_out_spat_gen)

        loss_disc_t = self.calc_loss_disc(disc_out_temp_true,
                                          disc_out_temp_gen)

        loss = None
        if train_gen:
            loss = loss_gen
        elif train_disc_s:
            loss = loss_disc_s
        elif train_disc_t:
            loss = loss_disc_t

        loss_details = {'loss_gen': loss_gen,
                        'loss_gen_content': loss_gen_content,
                        'loss_gen_advers_s': loss_gen_advers_s,
                        'loss_gen_advers_t': loss_gen_advers_t,
                        'loss_disc_s': loss_disc_s,
                        'loss_disc_t': loss_disc_t,
                        }

        return loss, loss_details

    def calc_val_loss(self, batch_handler, weight_gen_advers_s,
                      weight_gen_advers_t, loss_details):
        """Calculate the validation loss at the current state of model training

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        weight_gen_advers_s : float
            Weight factor for the adversarial loss component of the generator
            vs. the spatial discriminator.
        weight_gen_advers_t : float
            Weight factor for the adversarial loss component of the generator
            vs. the temporal discriminator.
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
                weight_gen_advers_s=weight_gen_advers_s,
                weight_gen_advers_t=weight_gen_advers_t,
                train_gen=False, train_disc_s=False, train_disc_t=False)

            loss_details = self.update_loss_details(loss_details,
                                                    v_loss_details,
                                                    len(val_batch),
                                                    prefix='val_')

        return loss_details

    def train_epoch(self, batch_handler, weight_gen_advers_s,
                    weight_gen_advers_t, train_gen,
                    train_disc_s, train_disc_t, disc_loss_bounds):
        """Train the GAN for one epoch.

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
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
        disc_loss_bounds : tuple
            Lower and upper bounds for the discriminator loss outside of which
            the discriminators will not train unless train_disc_s=True or
            train_disc_t=True and train_gen=False.

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components
        """
        disc_th_low = np.min(disc_loss_bounds)
        disc_th_high = np.max(disc_loss_bounds)
        loss_details = {'n_obs': 0, 'train_loss_disc_s': 0,
                        'train_loss_disc_t': 0}
        only_gen = train_gen and not (train_disc_s or train_disc_t)
        only_disc_s = train_disc_s and not (train_gen or train_disc_t)
        only_disc_t = train_disc_t and not (train_gen or train_disc_s)

        for ib, batch in enumerate(batch_handler):
            trained_gen = False
            trained_disc_s = False
            trained_disc_t = False
            b_loss_details = {}
            loss_disc_s = loss_details['train_loss_disc_s']
            loss_disc_t = loss_details['train_loss_disc_t']
            disc_s_too_good = loss_disc_s <= disc_th_low
            disc_t_too_good = loss_disc_t <= disc_th_low
            disc_s_too_bad = (loss_disc_s > disc_th_high) and train_disc_s
            disc_t_too_bad = (loss_disc_t > disc_th_high) and train_disc_t
            gen_too_good = disc_s_too_bad or disc_t_too_bad

            if only_gen or (train_gen and not gen_too_good):
                trained_gen = True
                b_loss_details = self.run_gradient_descent(
                    batch.low_res, batch.high_res, self.generator_weights,
                    weight_gen_advers_s=weight_gen_advers_s,
                    weight_gen_advers_t=weight_gen_advers_t,
                    train_gen=True, train_disc_s=False, train_disc_t=False)

            if only_disc_s or (train_disc_s and not disc_s_too_good):
                trained_disc_s = True
                b_loss_details = self.run_gradient_descent(
                    batch.low_res, batch.high_res, self.disc_spatial_weights,
                    weight_gen_advers_s=weight_gen_advers_s,
                    weight_gen_advers_t=weight_gen_advers_t,
                    train_gen=False, train_disc_s=True, train_disc_t=False)

            if only_disc_t or (train_disc_t and not disc_t_too_good):
                trained_disc_t = True
                b_loss_details = self.run_gradient_descent(
                    batch.low_res, batch.high_res, self.disc_temporal_weights,
                    weight_gen_advers_s=weight_gen_advers_s,
                    weight_gen_advers_t=weight_gen_advers_t,
                    train_gen=False, train_disc_s=False, train_disc_t=True)

            b_loss_details['gen_trained_frac'] = float(trained_gen)
            b_loss_details['disc_s_trained_frac'] = float(trained_disc_s)
            b_loss_details['disc_t_trained_frac'] = float(trained_disc_t)
            loss_details = self.update_loss_details(loss_details,
                                                    b_loss_details,
                                                    len(batch),
                                                    prefix='train_')

            logger.debug('Batch {} out of {} has epoch-average '
                         '(gen / disc_s / disc_t) loss of: '
                         '({:.2e} / {:.2e} / {:.2e}). '
                         'Trained (gen / disc_s / disc_t): ({} / {} / {})'
                         .format(ib, len(batch_handler),
                                 loss_details['train_loss_gen'],
                                 loss_details['train_loss_disc_s'],
                                 loss_details['train_loss_disc_t'],
                                 trained_gen, trained_disc_s, trained_disc_t))

            if all([trained_gen, trained_disc_s, trained_disc_t]):
                msg = ('For some reason none of the GAN networks trained '
                       'during batch {} out of {}!'
                       .format(ib, len(batch_handler)))
                logger.warning(msg)
                warn(msg)

        return loss_details

    def train(self, batch_handler, n_epoch,
              weight_gen_advers_s=0.001, weight_gen_advers_t=0.001,
              train_gen=True, train_disc_s=True, train_disc_t=True,
              disc_loss_bounds=(0.45, 0.6),
              checkpoint_int=None, out_dir='./spatial_gan_{epoch}',
              early_stop_on=None, early_stop_threshold=0.005,
              early_stop_n_epoch=5):
        """Train the GAN model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        n_epoch : int
            Number of epochs to train on
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
        logger.info('Training model with spatial/temporal '
                    'weight_gen_advers: {}/{} '
                    'for {} epochs starting at epoch {}'
                    .format(weight_gen_advers_s, weight_gen_advers_t,
                            n_epoch, epochs[0]))

        for epoch in epochs:
            loss_details = self.train_epoch(batch_handler, weight_gen_advers_s,
                                            weight_gen_advers_t, train_gen,
                                            train_disc_s, train_disc_t,
                                            disc_loss_bounds)

            loss_details = self.calc_val_loss(batch_handler,
                                              weight_gen_advers_s,
                                              weight_gen_advers_t,
                                              loss_details)

            logger.info('Epoch {} of {} '
                        'generator train/val loss: {:.2e}/{:.2e} '
                        'spatial discriminator train/val loss: {:.2e}/{:.2e} '
                        'temporal discriminator train/val loss: {:.2e}/{:.2e}'
                        .format(epoch, epochs[-1],
                                loss_details['train_loss_gen'],
                                loss_details['val_loss_gen'],
                                loss_details['train_loss_disc_s'],
                                loss_details['val_loss_disc_s'],
                                loss_details['train_loss_disc_t'],
                                loss_details['val_loss_disc_t'],
                                ))

            extras = {'weight_gen_advers_s': weight_gen_advers_s,
                      'weight_gen_advers_t': weight_gen_advers_t,
                      'disc_loss_bound_0': disc_loss_bounds[0],
                      'disc_loss_bound_1': disc_loss_bounds[1],
                      'learning_rate': self.optimizer_config['learning_rate']}
            stop = self.finish_epoch(epoch, epochs, t0, loss_details,
                                     checkpoint_int, out_dir,
                                     early_stop_on, early_stop_threshold,
                                     early_stop_n_epoch, extras=extras)
            if stop:
                break
