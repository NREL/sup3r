# -*- coding: utf-8 -*-
"""Sup3r model software"""
import copy
import os
import time
import json
import logging
import numpy as np
import pprint
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from rex.utilities.utilities import safe_json_load
from phygnn import CustomNetwork
from warnings import warn

from sup3r.models.abstract import AbstractSup3rGan
from sup3r.utilities import VERSION_RECORD
import sup3r.utilities.loss_metrics


logger = logging.getLogger(__name__)


class Sup3rGan(AbstractSup3rGan):
    """Basic sup3r GAN model."""

    def __init__(self, gen_layers, disc_layers, loss='MeanSquaredError',
                 optimizer=None, learning_rate=1e-4,
                 optimizer_disc=None, learning_rate_disc=None,
                 history=None, meta=None, means=None, stdevs=None, name=None):
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
            discriminative model (spatial or spatiotemporal discriminator). Can
            also be a str filepath to a .json config file containing the input
            layers argument or a .pkl for a saved pre-trained model.
        loss : str
            Loss function class name from sup3r.utilities.loss_metrics
            (prioritized) or tensorflow.keras.losses. Defaults to
            tf.keras.losses.MeanSquaredError.
        optimizer : tf.keras.optimizers.Optimizer | dict | None | str
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        optimizer_disc : tf.keras.optimizers.Optimizer | dict | None
            Same as optimizer input, but if specified this makes a different
            optimizer just for the discriminator network (spatial or
            spatiotemporal disc).
        learning_rate_disc : float, optional
            Same as learning_rate input, but if specified this makes a
            different learning_rate just for the discriminator network (spatial
            or spatiotemporal disc).
        history : pd.DataFrame | str | None
            Model training history with "epoch" index, str pointing to a saved
            history csv file with "epoch" as first column, or None for clean
            history
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

        self._version_record = VERSION_RECORD
        self.name = name if name is not None else self.__class__.__name__
        self._meta = meta if meta is not None else {}

        self.loss_name = loss
        self.loss_fun = self.get_loss_fun(loss)

        self._history = history
        if isinstance(self._history, str):
            self._history = pd.read_csv(self._history, index_col=0)

        optimizer_disc = optimizer_disc or copy.deepcopy(optimizer)
        learning_rate_disc = learning_rate_disc or learning_rate
        self._optimizer = self.init_optimizer(optimizer, learning_rate)
        self._optimizer_disc = self.init_optimizer(optimizer_disc,
                                                   learning_rate_disc)

        self._gen = self.load_network(gen_layers, 'generator')
        self._disc = self.load_network(disc_layers, 'discriminator')

        self._means = (means if means is None
                       else np.array(means).astype(np.float32))
        self._stdevs = (stdevs if stdevs is None
                        else np.array(stdevs).astype(np.float32))

    @staticmethod
    def init_optimizer(optimizer, learning_rate):
        """Initialize keras optimizer object.

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer | dict | None | str
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.

        Returns
        -------
        optimizer : tf.keras.optimizers.Optimizer
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
        config, or .pkl file saved pre-trained model.

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
            elif ('meta' in model
                  and f'config_{name}' in model['meta']
                  and 'hidden_layers' in model['meta'][f'config_{name}']):
                model = model['meta'][f'config_{name}']['hidden_layers']
            else:
                msg = ('Could not load model from json config, need '
                       '"hidden_layers" key or '
                       f'"meta/config_{name}/hidden_layers" '
                       ' at top level but only found: {}'
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
            os.makedirs(out_dir, exist_ok=True)

        fp_gen = os.path.join(out_dir, 'model_gen.pkl')
        self.generator.save(fp_gen)

        fp_disc = os.path.join(out_dir, 'model_disc.pkl')
        self.discriminator.save(fp_disc)

        fp_history = None
        if isinstance(self.history, pd.DataFrame):
            fp_history = os.path.join(out_dir, 'history.csv')
            self.history.to_csv(fp_history)

        self.save_params(out_dir)

        logger.info('Saved GAN to disk in directory: {}'.format(out_dir))

    @classmethod
    def load(cls, model_dir, verbose=True):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        model_dir : str
            Directory to load GAN model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : BaseModel
            Returns a pretrained gan model that was previously saved to out_dir
        """
        if verbose:
            logger.info('Loading GAN from disk in directory: {}'
                        .format(model_dir))
            msg = ('Active python environment versions: \n{}'
                   .format(pprint.pformat(VERSION_RECORD, indent=4)))
            logger.info(msg)

        fp_gen = os.path.join(model_dir, 'model_gen.pkl')
        fp_disc = os.path.join(model_dir, 'model_disc.pkl')
        params = cls.load_saved_params(model_dir, verbose=verbose)

        return cls(fp_gen, fp_disc, **params)

    @staticmethod
    def load_saved_params(out_dir, verbose=True):
        """Load saved GAN model_params (you need this and the gen+disc models
        to load a full GAN).

        Parameters
        ----------
        out_dir : str
            Directory to load GAN model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        params : dict
            Model parameters loaded from disk json file. This should be the
            same as self.model_params with and additional 'history' entry.
            Should be all the kwargs you need to init a GAN model.
        """

        fp_params = os.path.join(out_dir, 'model_params.json')
        with open(fp_params, 'r') as f:
            params = json.load(f)

        # using the saved model dir makes this more portable
        fp_history = os.path.join(out_dir, 'history.csv')
        if os.path.exists(fp_history):
            params['history'] = fp_history
        else:
            params['history'] = None

        if 'version_record' in params:
            version_record = params.pop('version_record')
            if verbose:
                logger.info('Loading GAN from disk that was created with the '
                            'following package versions: \n{}'
                            .format(pprint.pformat(version_record, indent=2)))

        return params

    @property
    def means(self):
        """Get the data normalization mean values.

        Returns
        -------
        np.ndarray
        """
        return self._means

    @property
    def stdevs(self):
        """Get the data normalization standard deviation values.

        Returns
        -------
        np.ndarray
        """
        return self._stdevs

    def set_norm_stats(self, new_means, new_stdevs):
        """Set the normalization statistics associated with a data batch
        handler to model attributes.

        Parameters
        ----------
        new_means : list | tuple | np.ndarray
            1D iterable of mean values with same length as number of features.
        new_stdevs : list | tuple | np.ndarray
            1D iterable of stdev values with same length as number of features.
        """

        if self._means is not None:
            logger.info('Setting new normalization statistics...')
            logger.info("Model's previous data mean values: {}"
                        .format(self._means))
            logger.info("Model's previous data stdev values: {}"
                        .format(self._stdevs))

        self._means = new_means
        self._stdevs = new_stdevs

        if not isinstance(self._means, np.ndarray):
            self._means = np.array(self._means)
        if not isinstance(self._stdevs, np.ndarray):
            self._stdevs = np.array(self._stdevs)

        logger.info('Set data normalization mean values: {}'
                    .format(self._means))
        logger.info('Set data normalization stdev values: {}'
                    .format(self._stdevs))

    def set_model_params(self, **kwargs):
        """Set parameters used for training the model

        Parameters
        ----------
        kwargs : dict
            Keyword arguments including 'training_features', 'output_features',
            'smoothed_features', 's_enhance', 't_enhance', 'smoothing'
        """

        keys = ('training_features', 'output_features', 'smoothed_features',
                's_enhance', 't_enhance', 'smoothing')
        keys = [k for k in keys if k in kwargs]

        for var in keys:
            val = getattr(self, var, None)
            if val is None:
                self.meta[var] = kwargs[var]
            elif val != kwargs[var]:
                msg = ('GAN was previously trained with {var}={} but '
                       'received new {var}={}'
                       .format(val, kwargs[var], var=var))
                logger.warning(msg)
                warn(msg)

    @staticmethod
    def get_loss_fun(loss):
        """Get the initialized loss function class from the sup3r loss library
        or the tensorflow losses.

        Parameters
        ----------
        loss : str
            Loss function class name from sup3r.utilities.loss_metrics
            (prioritized) or tensorflow.keras.losses. Defaults to
            tf.keras.losses.MeanSquaredError.

        Returns
        -------
        out : tf.keras.losses.Loss
            Initialized loss function class that is callable, e.g. if
            "MeanSquaredError" is requested, this will return
            an instance of tf.keras.losses.MeanSquaredError()
        """

        out = getattr(sup3r.utilities.loss_metrics, loss, None)
        if out is None:
            out = getattr(tf.keras.losses, loss, None)

        if out is None:
            msg = ('Could not find requested loss function "{}" in '
                   'sup3r.utilities.loss_metrics or tf.keras.losses.'
                   .format(loss))
            logger.error(msg)
            raise KeyError(msg)

        return out()

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

    def _needs_lr_exo(self, low_res):
        """Determine whether or not the sup3r model needs low-res exogenous
        data

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        needs_lr_exo : bool
            True if the model requires low-resolution exogenous data.
        """

        return low_res.shape[-1] < len(self.training_features)

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
            received normalized data with mean=0 stdev=1.
        un_norm_out : bool
           Flag to un-normalize synthetically generated output data to physical
           units
        exogenous_data : ndarray | None
            Exogenous data array, usually a 4D or 5D array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        exo_check = (exogenous_data is None or not self._needs_lr_exo(low_res))
        low_res = (low_res if exo_check
                   else np.concatenate((low_res, exogenous_data), axis=-1))

        if norm_in and self._means is not None:
            low_res = self.norm_input(low_res)

        hi_res = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
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

    @property
    def discriminator(self):
        """Get the discriminator model.

        Returns
        -------
        phygnn.base.CustomNetwork
        """
        return self._disc

    @property
    def discriminator_weights(self):
        """Get a list of layer weights and bias terms for the discriminator
        model.

        Returns
        -------
        list
        """
        return self.discriminator.weights

    def discriminate(self, hi_res, norm_in=False):
        """Run the discriminator model on a hi resolution input field.

        Parameters
        ----------
        hi_res : np.ndarray
            Real or fake high res data in a 4D or 5D tensor:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The disc should always
            received normalized data with mean=0 stdev=1.

        Returns
        -------
        out : np.ndarray
            Discriminator output logits
        """

        if isinstance(hi_res, tf.Tensor):
            hi_res = hi_res.numpy()

        if norm_in and self._means is not None:
            hi_res = hi_res if isinstance(hi_res, tf.Tensor) else hi_res.copy()
            for i, (m, s) in enumerate(zip(self._means, self._stdevs)):
                islice = tuple([slice(None)] * (len(hi_res.shape) - 1) + [i])
                hi_res[islice] = (hi_res[islice] - m) / s

        out = self.discriminator.layers[0](hi_res)
        for i, layer in enumerate(self.discriminator.layers[1:]):
            try:
                out = layer(out)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, out.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        out = out.numpy()

        return out

    @tf.function
    def _tf_discriminate(self, hi_res):
        """Run the discriminator model on a hi resolution input field.

        Parameters
        ----------
        hi_res : np.ndarray
            Real or fake high res data in a 4D or 5D tensor:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
            This input should always be normalized with mean=0 and stdev=1

        Returns
        -------
        out : np.ndarray
            Discriminator output logits
        """

        out = self.discriminator.layers[0](hi_res)
        for i, layer in enumerate(self.discriminator.layers[1:]):
            try:
                out = layer(out)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, out.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        return out

    def norm_input(self, low_res):
        """Normalize low resolution data being input to the generator.

        Parameters
        ----------
        low_res : np.ndarray
            Un-normalized low-resolution input data in physical units, usually
            a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        low_res : np.ndarray
            Normalized low-resolution input data, usually a 4D or 5D array of
            shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        if self._means is not None:
            if isinstance(low_res, tf.Tensor):
                low_res = low_res.numpy()

            low_res = low_res.copy()
            for idf in range(low_res.shape[-1]):
                low_res[..., idf] -= self._means[idf]

                if self._stdevs[idf] != 0:
                    low_res[..., idf] /= self._stdevs[idf]
                else:
                    msg = ('Standard deviation is zero for '
                           f'{self.training_features[idf]}')
                    logger.warning(msg)
                    warn(msg)

        return low_res

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

            for idf in range(hi_res.shape[-1]):
                feature_name = self.output_features[idf]
                i = self.training_features.index(feature_name)
                mean = self._means[i]
                stdev = self._stdevs[i]
                hi_res[..., idf] = (hi_res[..., idf] * stdev) + mean

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
        calculations for the generative network. This is functionally identical
        to optimizer_disc is no special optimizer model or learning rate was
        specified for the disc.

        Returns
        -------
        tf.keras.optimizers.Optimizer
        """
        return self._optimizer

    @property
    def optimizer_disc(self):
        """Get the tensorflow optimizer to perform gradient descent
        calculations for the discriminator network.

        Returns
        -------
        tf.keras.optimizers.Optimizer
        """
        return self._optimizer_disc

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

    def update_optimizer(self, option='generator', **kwargs):
        """Update optimizer by changing current configuration

        Parameters
        ----------
        option : str
            Which optimizer to update. Can be "generator", "discriminator", or
            "all"
        kwargs : dict
            kwargs to use for optimizer configuration update
        """

        if 'gen' in option.lower() or 'all' in option.lower():
            conf = self.get_optimizer_config(self.optimizer)
            conf.update(**kwargs)
            OptimizerClass = getattr(optimizers, conf['name'])
            self._optimizer = OptimizerClass.from_config(conf)

        if 'disc' in option.lower() or 'all' in option.lower():
            conf = self.get_optimizer_config(self.optimizer_disc)
            conf.update(**kwargs)
            OptimizerClass = getattr(optimizers, conf['name'])
            self._optimizer_disc = OptimizerClass.from_config(conf)

    @property
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

        if 'class' not in self._meta:
            self._meta['class'] = self.__class__.__name__

        return self._meta

    @property
    def smoothing(self):
        """Value of smoothing parameter used in gaussian filtering of coarsened
        high res data."""
        return self.meta.get('smoothing', None)

    @property
    def training_features(self):
        """Get the list of input feature names that the generative model was
        trained on."""
        return self.meta.get('training_features', None)

    @property
    def smoothed_features(self):
        """Get the list of smoothed input feature names that the generative
        model was trained on."""
        return self.meta.get('smoothed_features', None)

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

        config_optm_g = self.get_optimizer_config(self.optimizer)
        config_optm_d = self.get_optimizer_config(self.optimizer_disc)

        model_params = {'name': self.name,
                        'loss': self.loss_name,
                        'version_record': self.version_record,
                        'optimizer': config_optm_g,
                        'optimizer_disc': config_optm_d,
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
        generator and discriminator networks
        """
        return self.generator_weights + self.discriminator_weights

    @staticmethod
    def get_weight_update_fraction(history, comparison_key,
                                   update_bounds=(0.5, 0.95),
                                   update_frac=0.0):
        """Get the factor by which to multiply previous adversarial loss
        weight

        Parameters
        ----------
        history : dict
            Dictionary with information on how often discriminators
            were trained during previous epoch.
        comparison_key : str
            history key to use for update check
        update_bounds : tuple
            Tuple specifying allowed range for history[comparison_key]. If
            history[comparison_key] < update_bounds[0] then the weight will be
            increased by (1 + update_frac). If history[comparison_key] >
            update_bounds[1] then the weight will be decreased by 1 / (1 +
            update_frac).
        update_frac : float
            Fraction by which to increase/decrease adversarial loss weight

        Returns
        -------
        float
            Factor by which to multiply old weight to get updated weight
        """

        val = history[comparison_key]
        if isinstance(val, (list, tuple, np.ndarray)):
            val = val[-1]

        if val < update_bounds[0]:
            return 1 + update_frac
        elif val > update_bounds[1]:
            return 1 / (1 + update_frac)
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
                if isinstance(v, str):
                    msg_format = '\t{}: {}'
                else:
                    msg_format = '\t{}: {:.2e}'
                if level.lower() == 'info':
                    logger.info(msg_format.format(k, v))
                else:
                    logger.debug(msg_format.format(k, v))

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

    @tf.function
    def calc_loss_gen_content(self, hi_res_true, hi_res_gen):
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

        loss_gen_content = self.loss_fun(hi_res_true, hi_res_gen)

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

    def calc_loss(self, hi_res_true, hi_res_gen, weight_gen_advers=0.001,
                  train_gen=True, train_disc=False):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_gen : bool
            True if generator is being trained, then loss=loss_gen
        train_disc : bool
            True if disc is being trained, then loss=loss_disc

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the network being trained
            (either generator or one of the discriminators)
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        if hi_res_gen.shape != hi_res_true.shape:
            msg = ('The tensor shapes of the synthetic output {} and '
                   'true high res {} did not have matching shape! '
                   'Check the spatiotemporal enhancement multipliers in your '
                   'your model config and data handlers.'
                   .format(hi_res_gen.shape, hi_res_true.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        disc_out_true = self._tf_discriminate(hi_res_true)
        disc_out_gen = self._tf_discriminate(hi_res_gen)

        loss_gen_content = self.calc_loss_gen_content(hi_res_true, hi_res_gen)
        loss_gen_advers = self.calc_loss_gen_advers(disc_out_gen)
        loss_gen = (loss_gen_content + weight_gen_advers * loss_gen_advers)

        loss_disc = self.calc_loss_disc(disc_out_true, disc_out_gen)

        loss = None
        if train_gen:
            loss = loss_gen
        elif train_disc:
            loss = loss_disc

        loss_details = {'loss_gen': loss_gen,
                        'loss_gen_content': loss_gen_content,
                        'loss_gen_advers': loss_gen_advers,
                        'loss_disc': loss_disc,
                        }

        return loss, loss_details

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
            high_res_gen = self._tf_generate(val_batch.low_res)
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
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
        disc_loss_bounds : tuple
            Lower and upper bounds for the discriminator loss outside of which
            the discriminators will not train unless train_disc=True or
            and train_gen=False.

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
            disc_too_good = loss_disc <= disc_th_low
            disc_too_bad = (loss_disc > disc_th_high) and train_disc
            gen_too_good = disc_too_bad

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
                    batch.low_res, batch.high_res, self.discriminator_weights,
                    weight_gen_advers=weight_gen_advers,
                    optimizer=self.optimizer_disc,
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

            if all([not trained_gen, not trained_disc]):
                msg = ('For some reason none of the GAN networks trained '
                       'during batch {} out of {}!'
                       .format(ib, len(batch_handler)))
                logger.warning(msg)
                warn(msg)

        return loss_details

    def update_adversarial_weights(self, history, adaptive_update_fraction,
                                   adaptive_update_bounds,
                                   weight_gen_advers,
                                   train_disc):
        """Update spatial / temporal adversarial loss weights based on training
        fraction history.

        Parameters
        ----------
        history : dict
            Dictionary with information on how often discriminators
            were trained during current and previous epochs.
        adaptive_update_fraction : float
            Amount by which to increase or decrease adversarial loss weights
            for adaptive updates
        adaptive_update_bounds : tuple
            Tuple specifying allowed range for history[comparison_key]. If
            history[comparison_key] < update_bounds[0] then the weight will be
            increased by (1 + update_frac). If history[comparison_key] >
            update_bounds[1] then the weight will be decreased by 1 / (1 +
            update_frac).
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_disc : bool
            Whether the discriminator was set to be trained during the
            previous epoch

        Returns
        -------
        weight_gen_advers : float
            Updated weight factor for the adversarial loss component of the
            generator vs. the discriminator.
        """

        if adaptive_update_fraction > 0:
            update_frac = 1
            if train_disc:
                update_frac = self.get_weight_update_fraction(
                    history, 'train_disc_trained_frac',
                    update_frac=adaptive_update_fraction,
                    update_bounds=adaptive_update_bounds)
                weight_gen_advers *= update_frac

            if update_frac != 1:
                logger.debug(
                    f'New discriminator weight: {weight_gen_advers:.3f}')

        return weight_gen_advers

    def train(self, batch_handler, n_epoch,
              weight_gen_advers=0.001,
              train_gen=True,
              train_disc=True,
              disc_loss_bounds=(0.45, 0.6),
              checkpoint_int=None,
              out_dir='./gan_{epoch}',
              early_stop_on=None,
              early_stop_threshold=0.005,
              early_stop_n_epoch=5,
              adaptive_update_bounds=(0.9, 0.99),
              adaptive_update_fraction=0.0):
        """Train the GAN model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        n_epoch : int
            Number of epochs to train on
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
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
        adaptive_update_bounds : tuple
            Tuple specifying allowed range for loss_details[comparison_key]. If
            history[comparison_key] < threshold_range[0] then the weight will
            be increased by (1 + update_frac). If history[comparison_key] >
            threshold_range[1] then the weight will be decreased by 1 / (1 +
            update_frac).
        adaptive_update_fraction : float
            Amount by which to increase or decrease adversarial weights for
            adaptive updates
        """

        self.set_norm_stats(batch_handler.means, batch_handler.stds)
        self.set_model_params(
            s_enhance=batch_handler.s_enhance,
            t_enhance=batch_handler.t_enhance,
            smoothing=batch_handler.smoothing,
            training_features=batch_handler.training_features,
            output_features=batch_handler.output_features,
            smoothed_features=batch_handler.smoothed_features)

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(
                columns=['elapsed_time'])
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        t0 = time.time()
        logger.info('Training model with adversarial weight: {} '
                    'for {} epochs starting at epoch {}'
                    .format(weight_gen_advers, n_epoch, epochs[0]))

        for epoch in epochs:
            loss_details = self.train_epoch(batch_handler, weight_gen_advers,
                                            train_gen, train_disc,
                                            disc_loss_bounds)

            loss_details = self.calc_val_loss(batch_handler, weight_gen_advers,
                                              loss_details)

            msg = f'Epoch {epoch} of {epochs[-1]} '
            msg += 'gen/disc train loss: {:.2e}/{:.2e} '.format(
                loss_details["train_loss_gen"],
                loss_details["train_loss_disc"])

            if all(loss in loss_details for loss
                   in ('val_loss_gen', 'val_loss_disc')):
                msg += 'gen/disc val loss: {:.2e}/{:.2e} '.format(
                    loss_details["val_loss_gen"],
                    loss_details["val_loss_disc"])

            logger.info(msg)

            lr_g = self.get_optimizer_config(self.optimizer)['learning_rate']
            lr_d = self.get_optimizer_config(
                self.optimizer_disc)['learning_rate']

            extras = {'weight_gen_advers': weight_gen_advers,
                      'disc_loss_bound_0': disc_loss_bounds[0],
                      'disc_loss_bound_1': disc_loss_bounds[1],
                      'learning_rate_gen': lr_g,
                      'learning_rate_disc': lr_d}

            weight_gen_advers = self.update_adversarial_weights(
                loss_details, adaptive_update_fraction, adaptive_update_bounds,
                weight_gen_advers, train_disc)

            stop = self.finish_epoch(epoch, epochs, t0, loss_details,
                                     checkpoint_int, out_dir,
                                     early_stop_on, early_stop_threshold,
                                     early_stop_n_epoch, extras=extras)

            if stop:
                break
