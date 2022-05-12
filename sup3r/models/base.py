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

    def __init__(self, optimizer=None,
                 learning_rate=1e-4,
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

        self._optimizer = self.init_optimizer(optimizer, learning_rate)

    @staticmethod
    def init_optimizer(optimizer, learning_rate):
        """Initialize keras optimizer object.

        Parameters
        ----------
        optimizer : tensorflow.keras.optimizers | dict | None | str
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.

        Returns
        -------
        optimizer : tensorflow.keras.optimizers.Optimizer
            Initialized optimizer object.
        """
        if isinstance(optimizer, dict):
            class_name = optimizer['name']
            OptimizerClass = getattr(optimizers, class_name)
            optimizer = OptimizerClass.from_config(optimizer)
        elif optimizer is None:
            optimizer = optimizers.Adam(learning_rate=learning_rate)

        return optimizer

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

    def generate(self, low_res, norm_in=True, un_norm_out=True):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data
        """

        if norm_in and self._means is not None:
            if isinstance(low_res, tf.Tensor):
                msg = 'Cannot normalize tensor input, must be array'
                logger.error(msg)
                raise TypeError(msg)

            low_res = low_res.copy()
            for i, (m, s) in enumerate(zip(self._means, self._stdevs)):
                low_res[..., i] -= m
                if s > 0:
                    low_res[..., i] /= s
                else:
                    logger.warning('Standard deviation is zero for '
                                   f'{self.training_features[i]}')

        hi_res = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
                hi_res = layer(hi_res)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, hi_res.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        if isinstance(hi_res, tf.Tensor):
            hi_res = hi_res.numpy()

        if un_norm_out:
            if self._means is not None:
                for i, feature in enumerate(self.training_features):
                    if feature in self.output_features:
                        m = self._means[i]
                        s = self._stdevs[i]
                        j = self.output_features.index(feature)
                        hi_res[..., j] = (hi_res[..., j] * s) + m

        return hi_res

    @tf.function
    def _tf_generate(self, low_res):
        """Use the generator model to generate high res data from los res input

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data. The generator should always
            received normalized data with mean=0 stdev=1.

        Returns
        -------
        hi_res : tf.Tensor
            Synthetically generated high-resolution data
        """

        hi_res = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
                hi_res = layer(hi_res)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, hi_res.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

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

    @staticmethod
    def get_optimizer_config(optimizer):
        """Get a config that defines the current GAN optimizer

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
            TF-Keras optimizer object

        Returns
        -------
        config : dict
            Optimizer config
        """
        conf = optimizer.get_config()
        for k, v in conf.items():
            # need to convert numpy dtypes to float/int for json.dump()
            if np.issubdtype(type(v), np.floating):
                conf[k] = float(v)
            elif np.issubdtype(type(v), np.integer):
                conf[k] = int(v)
        return conf

    @abstractmethod
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
                        'optimizer': self.get_optimizer_config(self.optimizer),
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
    def get_adversarial_weight_update_fraction(loss_details, comparison_key,
                                               threshold_range=(0.5, 0.95),
                                               update_frac=0.025):
        """Get the factor by which to multiply previous adversarial loss
        weight

        Parameters
        ----------
        loss_details : dict
            Dictionary with information on how often discriminators
            were trained and other history information.
        comparison_key : str
            loss_details key to use for update check
        threshold_range : tuple
            Tuple specifying allowed range for loss_details[comparison_key]. If
            loss_details[comparison_key] < threshold_range[0] then the weight
            will be increased by (1 + update_frac). If
            loss_details[comparison_key] > threshold_range[1] then the weight
            will be decreased by (1 - update_frac).
        update_frac : float
            Fraction by which to increase/decrease adversarial loss weight

        Returns
        -------
        float
            Factor by which to multiply old weight to get updated weight
        """

        trained_frac = loss_details[comparison_key]

        if trained_frac < threshold_range[0]:
            return 1 + update_frac
        elif trained_frac > threshold_range[1]:
            return 1 - update_frac
        else:
            return 1

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

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(training_weights)

            hi_res_gen = self._tf_generate(low_res)
            loss_out = self.calc_loss(hi_res_true, hi_res_gen,
                                      **calc_loss_kwargs)
            loss, loss_details = loss_out

            grad = tape.gradient(loss, training_weights)

        if optimizer is None:
            optimizer = self.optimizer

        optimizer.apply_gradients(zip(grad, training_weights))

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
