# -*- coding: utf-8 -*-
"""
Abstract class to define the required interface for Sup3r model subclasses
"""
import os
import json
from abc import ABC, abstractmethod
from phygnn import CustomNetwork
from sup3r.utilities import VERSION_RECORD
import sup3r.utilities.loss_metrics
import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import logging
import pprint
from warnings import warn

logger = logging.getLogger(__name__)


class AbstractSup3rGan(ABC):
    """
    Abstract class to define the required interface for Sup3r model subclasses

    Note that this only sets the required interfaces for a GAN that can be
    loaded from disk and used to predict synthetic outputs. The interface for
    models that can be trained will be set in another class.
    """

    @classmethod
    @abstractmethod
    def load(cls, model_dir, verbose=True):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        model_dir
            Directory to load GAN model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : BaseModel
            Returns a pretrained gan model that was previously saved to
            model_dir
        """

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

    @abstractmethod
    def generate(self, low_res):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
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

    @property
    def s_enhance(self):
        """Factor by which model will enhance spatial resolution. Used in
        model training during high res coarsening"""
        return self.meta.get('s_enhance', None)

    @property
    def t_enhance(self):
        """Factor by which model will enhance temporal resolution. Used in
        model training during high res coarsening"""
        return self.meta.get('t_enhance', None)

    @property
    @abstractmethod
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

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
    def smoothing(self):
        """Value of smoothing parameter used in gaussian filtering of coarsened
        high res data."""
        return self.meta.get('smoothing', None)

    @property
    def smoothed_features(self):
        """Get the list of smoothed input feature names that the generative
        model was trained on."""
        return self.meta.get('smoothed_features', None)

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        model_params = {'meta': self.meta}
        return model_params

    @property
    def version_record(self):
        """A record of important versions that this model was built with.

        Returns
        -------
        dict
        """
        return VERSION_RECORD

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
                msg = ('Model was previously trained with {var}={} but '
                       'received new {var}={}'
                       .format(val, kwargs[var], var=var))
                logger.warning(msg)
                warn(msg)

    def save_params(self, out_dir):
        """
        Parameters
        ----------
        out_dir : str
            Directory to save linear model params. This directory will be
            created if it does not already exist.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        fp_params = os.path.join(out_dir, 'model_params.json')
        with open(fp_params, 'w') as f:
            params = self.model_params
            json.dump(params, f, sort_keys=True, indent=2)


class AbstractSup3rGanTraining(ABC):
    """
    Abstract class to define the required training interface
    for Sup3r model subclasses
    """

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

    @staticmethod
    def load_saved_params(out_dir, verbose=True):
        """Load saved model_params (you need this and the gen+disc models
        to load a full model).

        Parameters
        ----------
        out_dir : str
            Directory to load model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        params : dict
            Model parameters loaded from disk json file. This should be the
            same as self.model_params with and additional 'history' entry.
            Should be all the kwargs you need to init a model.
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
                logger.info('Loading model from disk '
                            'that was created with the '
                            'following package versions: \n{}'
                            .format(pprint.pformat(version_record, indent=2)))

        return params

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
    @abstractmethod
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

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

    @staticmethod
    def get_optimizer_config(optimizer):
        """Get a config that defines the current model optimizer

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
