"""Abstract class defining the required interface for Sup3r model subclasses"""

import copy
import json
import locale
import logging
import os
import pprint
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from warnings import warn

import numpy as np
import pandas as pd
import tensorflow as tf
from phygnn import CustomNetwork
from phygnn.layers.custom_layers import Sup3rAdder, Sup3rConcat
from rex.utilities.utilities import safe_json_load
from tensorflow.keras import optimizers

import sup3r.utilities.loss_metrics
from sup3r.preprocessing.data_handlers import ExoData
from sup3r.preprocessing.utilities import numpy_if_tensor
from sup3r.utilities import VERSION_RECORD
from sup3r.utilities.utilities import Timer, camel_to_underscore, safe_cast

logger = logging.getLogger(__name__)


class TensorboardMixIn:
    """MixIn class for tensorboard logging and profiling.

    Note: To monitor model training with tensorboard run ``ssh -N -f -L
    localhost:6006:localhost:6006 <user>@<host>`` on your local machine and
    then ``tensorboard --logdir <log_dir>`` on the remote machine
    """

    def __init__(self):
        self._tb_writer = None
        self._tb_log_dir = None
        self._write_tb_profile = False
        self._total_batches = None
        self._history = None
        self.timer = Timer()

    @property
    def total_batches(self):
        """Record of total number of batches for logging."""
        if self._total_batches is None and self._history is None:
            self._total_batches = 0
        elif self._history is None and 'total_batches' in self._history:
            self._total_batches = self._history['total_batches'].values[-1]
        elif self._total_batches is None and self._history is not None:
            self._total_batches = 0
        return self._total_batches

    @total_batches.setter
    def total_batches(self, value):
        """Set total number of batches."""
        self._total_batches = value

    def dict_to_tensorboard(self, entry):
        """Write data to tensorboard log file. This is usually a loss_details
        dictionary.

        Parameters
        ----------
        entry: dict
            Dictionary of values to write to tensorboard log file
        """
        if self._tb_writer is not None:
            with self._tb_writer.as_default():
                for name, value in entry.items():
                    if isinstance(value, str):
                        tf.summary.text(name, value, self.total_batches)
                    else:
                        tf.summary.scalar(name, value, self.total_batches)

    def profile_to_tensorboard(self, name):
        """Write profile data to tensorboard log file.

        Parameters
        ----------
        name : str
            Tag name to use for profile info
        """
        if self._tb_writer is not None and self._write_tb_profile:
            with self._tb_writer.as_default():
                tf.summary.trace_export(
                    name=name,
                    step=self.total_batches,
                    profiler_outdir=self._tb_log_dir,
                )

    def _init_tensorboard_writer(self, out_dir):
        """Initialize the ``tf.summary.SummaryWriter`` to use for writing
        tensorboard compatible log files.

        Parameters
        ----------
        out_dir : str
            Standard out_dir where model epochs are saved. e.g. './gan_{epoch}'
        """
        tb_log_pardir = os.path.abspath(os.path.join(out_dir, os.pardir))
        self._tb_log_dir = os.path.join(tb_log_pardir, 'logs')
        os.makedirs(self._tb_log_dir, exist_ok=True)
        self._tb_writer = tf.summary.create_file_writer(self._tb_log_dir)


class AbstractInterface(ABC):
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

    @abstractmethod
    def generate(
        self, low_res, norm_in=True, un_norm_out=True, exogenous_data=None
    ):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function."""

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
    def input_dims(self):
        """Get dimension of model generator input. This is usually 4D for
        spatial models and 5D for spatiotemporal models. This gives the input
        to the first step if the model is multi-step. Returns 5 for linear
        models.

        Returns
        -------
        int
        """
        # pylint: disable=E1101
        if hasattr(self, '_gen'):
            return self._gen.layers[0].rank
        if hasattr(self, 'models'):
            return self.models[0].input_dims
        return 5

    @property
    def is_5d(self):
        """Check if model expects spatiotemporal input"""
        return self.input_dims == 5

    @property
    def is_4d(self):
        """Check if model expects spatial only input"""
        return self.input_dims == 4

    # pylint: disable=E1101
    def get_s_enhance_from_layers(self):
        """Compute factor by which model will enhance spatial resolution from
        layer attributes. Used in model training during high res coarsening"""
        s_enhance = None
        if hasattr(self, '_gen'):
            s_enhancements = [
                getattr(layer, '_spatial_mult', 1)
                for layer in self._gen.layers
            ]
            s_enhance = int(np.prod(s_enhancements))
        return s_enhance

    # pylint: disable=E1101
    def get_t_enhance_from_layers(self):
        """Compute factor by which model will enhance temporal resolution from
        layer attributes. Used in model training during high res coarsening"""
        t_enhance = None
        if hasattr(self, '_gen'):
            t_enhancements = [
                getattr(layer, '_temporal_mult', 1)
                for layer in self._gen.layers
            ]
            t_enhance = int(np.prod(t_enhancements))
        return t_enhance

    @property
    def s_enhance(self):
        """Factor by which model will enhance spatial resolution. Used in
        model training during high res coarsening and also in forward pass
        routine to determine shape of needed exogenous data"""
        models = getattr(self, 'models', [self])
        s_enhances = [m.meta.get('s_enhance', None) for m in models]
        s_enhance = (
            self.get_s_enhance_from_layers()
            if any(s is None for s in s_enhances)
            else int(np.prod(s_enhances))
        )
        if len(models) == 1:
            self.meta['s_enhance'] = s_enhance
        return s_enhance

    @property
    def t_enhance(self):
        """Factor by which model will enhance temporal resolution. Used in
        model training during high res coarsening and also in forward pass
        routine to determine shape of needed exogenous data"""
        models = getattr(self, 'models', [self])
        t_enhances = [m.meta.get('t_enhance', None) for m in models]
        t_enhance = (
            self.get_t_enhance_from_layers()
            if any(t is None for t in t_enhances)
            else int(np.prod(t_enhances))
        )
        if len(models) == 1:
            self.meta['t_enhance'] = t_enhance
        return t_enhance

    @property
    def s_enhancements(self):
        """List of spatial enhancement factors. In the case of a single step
        model this is just ``[self.s_enhance]``. This is used to determine
        shapes of needed exogenous data in forward pass routine"""
        if hasattr(self, 'models'):
            return [model.s_enhance for model in self.models]
        return [self.s_enhance]

    @property
    def t_enhancements(self):
        """List of temporal enhancement factors. In the case of a single step
        model this is just ``[self.t_enhance]``. This is used to determine
        shapes of needed exogenous data in forward pass routine"""
        if hasattr(self, 'models'):
            return [model.t_enhance for model in self.models]
        return [self.t_enhance]

    @property
    def input_resolution(self):
        """Resolution of input data. Given as a dictionary
        ``{'spatial': '...km', 'temporal': '...min'}``. The numbers are
        required to be integers in the units specified. The units are not
        strict as long as the resolution of the exogenous data, when extracting
        exogenous data, is specified in the same units."""
        input_resolution = self.meta.get('input_resolution', None)
        msg = 'model.input_resolution is None. This needs to be set.'
        assert input_resolution is not None, msg
        return input_resolution

    def _get_numerical_resolutions(self):
        """Get the input and output resolutions without units. e.g. for
        ``{"spatial": "30km", "temporal": "60min"}`` this returns
        ``{"spatial": 30, "temporal": 60}``"""
        ires_num = {
            k: int(re.search(r'\d+', v).group(0))
            for k, v in self.input_resolution.items()
        }
        enhancements = {'spatial': self.s_enhance, 'temporal': self.t_enhance}
        ores_num = {k: v // enhancements[k] for k, v in ires_num.items()}
        return ires_num, ores_num

    def _ensure_valid_input_resolution(self):
        """Ensure ehancement factors evenly divide input_resolution"""

        if self.input_resolution is None:
            return

        ires_num, ores_num = self._get_numerical_resolutions()
        s_enhance = self.meta['s_enhance']
        t_enhance = self.meta['t_enhance']
        check = (
            ires_num['temporal'] / ores_num['temporal'] == t_enhance
            and ires_num['spatial'] / ores_num['spatial'] == s_enhance
        )
        msg = (
            f'Enhancement factors (s_enhance={s_enhance}, '
            f't_enhance={t_enhance}) do not evenly divide '
            f'input resolution ({self.input_resolution})'
        )
        if not check:
            logger.error(msg)
            raise RuntimeError(msg)

    def _ensure_valid_enhancement_factors(self):
        """Ensure user provided enhancement factors are the same as those
        computed from layer attributes"""
        t_enhance = self.meta.get('t_enhance', None)
        s_enhance = self.meta.get('s_enhance', None)
        if s_enhance is None or t_enhance is None:
            return

        layer_se = self.get_s_enhance_from_layers()
        layer_te = self.get_t_enhance_from_layers()
        layer_se = layer_se if layer_se is not None else self.meta['s_enhance']
        layer_te = layer_te if layer_te is not None else self.meta['t_enhance']
        msg = (
            f'Enhancement factors computed from layer attributes '
            f'(s_enhance={layer_se}, t_enhance={layer_te}) '
            f'conflict with user provided values (s_enhance={s_enhance}, '
            f't_enhance={t_enhance})'
        )
        check = layer_se == s_enhance or layer_te == t_enhance
        if not check:
            logger.error(msg)
            raise RuntimeError(msg)

    @property
    def output_resolution(self):
        """Resolution of output data. Given as a dictionary
        {'spatial': '...km', 'temporal': '...min'}. This is computed from the
        input resolution and the enhancement factors."""
        output_res = self.meta.get('output_resolution', None)
        if self.input_resolution is not None and output_res is None:
            ires_num, ores_num = self._get_numerical_resolutions()
            output_res = {
                k: v.replace(str(ires_num[k]), str(ores_num[k]))
                for k, v in self.input_resolution.items()
            }
            self.meta['output_resolution'] = output_res
        return output_res

    def _combine_fwp_input(self, low_res, exogenous_data=None):
        """Combine exogenous_data at input resolution with low_res data prior
        to forward pass through generator

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        exogenous_data : dict | ExoData | None
            Special dictionary (class:`ExoData`) of exogenous feature data with
            entries describing whether features should be combined at input, a
            mid network layer, or with output. This doesn't have to include
            the 'model' key since this data is for a single step model.

        Returns
        -------
        low_res : np.ndarray
            Low-resolution input data combined with exogenous_data, usually a
            4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        if exogenous_data is None:
            return low_res

        if (
            not isinstance(exogenous_data, ExoData)
            and exogenous_data is not None
        ):
            exogenous_data = ExoData(exogenous_data)

        fnum_diff = len(self.lr_features) - low_res.shape[-1]
        exo_feats = [] if fnum_diff <= 0 else self.lr_features[-fnum_diff:]
        msg = (
            f'Provided exogenous_data: {exogenous_data} is missing some '
            f'required features ({exo_feats})'
        )
        assert all(feature in exogenous_data for feature in exo_feats), msg
        if exogenous_data is not None and fnum_diff > 0:
            for feature in exo_feats:
                exo_input = exogenous_data.get_combine_type_data(
                    feature, 'input'
                )
                if exo_input is not None:
                    low_res = np.concatenate((low_res, exo_input), axis=-1)

        return low_res

    def _combine_fwp_output(self, hi_res, exogenous_data=None):
        """Combine exogenous_data at output resolution with generated hi_res
        data following forward pass output.

        Parameters
        ----------
        hi_res : np.ndarray
            High-resolution output data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        exogenous_data : dict | ExoData | None
            Special dictionary (class:`ExoData`) of exogenous feature data with
            entries describing whether features should be combined at input, a
            mid network layer, or with output. This doesn't have to include
            the 'model' key since this data is for a single step model.

        Returns
        -------
        hi_res : np.ndarray
            High-resolution output data combined with exogenous_data, usually a
            4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        if exogenous_data is None:
            return hi_res

        if (
            not isinstance(exogenous_data, ExoData)
            and exogenous_data is not None
        ):
            exogenous_data = ExoData(exogenous_data)

        fnum_diff = len(self.hr_out_features) - hi_res.shape[-1]
        exo_feats = [] if fnum_diff <= 0 else self.hr_out_features[-fnum_diff:]
        msg = (
            'Provided exogenous_data is missing some required features '
            f'({exo_feats})'
        )
        assert all(feature in exogenous_data for feature in exo_feats), msg
        if exogenous_data is not None and fnum_diff > 0:
            for feature in exo_feats:
                exo_output = exogenous_data.get_combine_type_data(
                    feature, 'output'
                )
                if exo_output is not None:
                    hi_res = np.concatenate((hi_res, exo_output), axis=-1)
        return hi_res

    @tf.function
    def _combine_loss_input(self, high_res_true, high_res_gen):
        """Combine exogenous feature data from high_res_true with high_res_gen
        for loss calculation

        Parameters
        ----------
        high_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        high_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.

        Returns
        -------
        high_res_gen : tf.Tensor
            Same as input with exogenous data combined with high_res input
        """
        if high_res_true.shape[-1] > high_res_gen.shape[-1]:
            for feature in self.hr_exo_features:
                f_idx = self.hr_exo_features.index(feature)
                f_idx += len(self.hr_out_features)
                exo_data = high_res_true[..., f_idx : f_idx + 1]
                high_res_gen = tf.concat((high_res_gen, exo_data), axis=-1)
        return high_res_gen

    @property
    @abstractmethod
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

    @property
    def lr_features(self):
        """Get a list of low-resolution features input to the generative model.
        This includes low-resolution features that might be supplied
        exogenously at inference time but that were in the low-res batches
        during training"""
        return self.meta.get('lr_features', [])

    @property
    def hr_out_features(self):
        """Get the list of high-resolution output feature names that the
        generative model outputs."""
        return self.meta.get('hr_out_features', [])

    @property
    def hr_exo_features(self):
        """Get list of high-resolution exogenous filter names the model uses.
        If the model has N concat or add layers this list will be the last N
        features in the training features list. The ordering is assumed to be
        the same as the order of concat or add layers. If training features is
        [..., topo, sza], and the model has 2 concat or add layers, exo
        features will be [topo, sza]. Topo will then be used in the first
        concat layer and sza will be used in the second"""
        # pylint: disable=E1101
        features = []
        if hasattr(self, '_gen'):
            features = [
                layer.name
                for layer in self._gen.layers
                if isinstance(layer, (Sup3rAdder, Sup3rConcat))
            ]
        return features

    @property
    def smoothing(self):
        """Value of smoothing parameter used in gaussian filtering of coarsened
        high res data."""
        return self.meta.get('smoothing', None)

    @property
    def smoothed_features(self):
        """Get the list of smoothed input feature names that the generative
        model was trained on."""
        return self.meta.get('smoothed_features', [])

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        return {'meta': self.meta}

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
            Keyword arguments including 'input_resolution',
            'lr_features', 'hr_exo_features', 'hr_out_features',
            'smoothed_features', 's_enhance', 't_enhance', 'smoothing'
        """

        keys = (
            'input_resolution',
            'lr_features',
            'hr_exo_features',
            'hr_out_features',
            'smoothed_features',
            's_enhance',
            't_enhance',
            'smoothing',
        )
        keys = [k for k in keys if k in kwargs]

        hr_exo_feat = kwargs.get('hr_exo_features', [])
        msg = (
            f'Expected high-res exo features {self.hr_exo_features} '
            f'based on model architecture but received "hr_exo_features" '
            f'from data handler: {hr_exo_feat}'
        )
        assert list(self.hr_exo_features) == list(hr_exo_feat), msg

        for var in keys:
            val = self.meta.get(var, None)
            if val is None:
                self.meta[var] = kwargs[var]
            elif val != kwargs[var]:
                msg = (
                    'Model was previously trained with {var}={} but '
                    'received new {var}={}'.format(val, kwargs[var], var=var)
                )
                logger.warning(msg)
                warn(msg)

        self._ensure_valid_enhancement_factors()
        self._ensure_valid_input_resolution()

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
        with open(
            fp_params, 'w', encoding=locale.getpreferredencoding(False)
        ) as f:
            params = self.model_params
            json.dump(params, f, sort_keys=True, indent=2, default=safe_cast)


# pylint: disable=E1101,W0201,E0203
class AbstractSingleModel(ABC, TensorboardMixIn):
    """
    Abstract class to define the required training interface
    for Sup3r model subclasses
    """

    def __init__(self):
        super().__init__()
        self.gpu_list = tf.config.list_physical_devices('GPU')
        self.default_device = '/cpu:0' if len(self.gpu_list) == 0 else '/gpu:0'
        self._version_record = VERSION_RECORD
        self.name = None
        self._meta = None
        self.loss_name = None
        self.loss_fun = None
        self._history = None
        self._optimizer = None
        self._gen = None
        self._means = None
        self._stdevs = None
        self._train_record = pd.DataFrame()
        self._val_record = pd.DataFrame()

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
            elif (
                'meta' in model
                and f'config_{name}' in model['meta']
                and 'hidden_layers' in model['meta'][f'config_{name}']
            ):
                model = model['meta'][f'config_{name}']['hidden_layers']
            else:
                msg = (
                    'Could not load model from json config, need '
                    '"hidden_layers" key or '
                    f'"meta/config_{name}/hidden_layers" '
                    ' at top level but only found: {}'.format(model.keys())
                )
                logger.error(msg)
                raise KeyError(msg)

        elif isinstance(model, str) and model.endswith('.pkl'):
            with tf.device(self.default_device):
                model = CustomNetwork.load(model)

        if isinstance(model, list):
            model = CustomNetwork(hidden_layers=model, name=name)

        if not isinstance(model, CustomNetwork):
            msg = (
                'Something went wrong. Tried to load a custom network '
                'but ended up with a model of type "{}"'.format(type(model))
            )
            logger.error(msg)
            raise TypeError(msg)

        return model

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
        new_means : dict | None
            Set of mean values for data normalization keyed by feature name.
            Can be used to maintain a consistent normalization scheme between
            transfer learning domains.
        new_stdevs : dict | None
            Set of stdev values for data normalization keyed by feature name.
            Can be used to maintain a consistent normalization scheme between
            transfer learning domains.
        """

        if new_means is not None and new_stdevs is not None:
            logger.info('Setting new normalization statistics...')
            logger.info(
                "Model's previous data mean values:\n%s",
                pprint.pformat(self._means, indent=2),
            )
            logger.info(
                "Model's previous data stdev values:\n%s",
                pprint.pformat(self._stdevs, indent=2),
            )

            self._means = {k: np.float32(v) for k, v in new_means.items()}
            self._stdevs = {k: np.float32(v) for k, v in new_stdevs.items()}

            if not isinstance(self._means, dict) or not isinstance(
                self._stdevs, dict
            ):
                msg = (
                    'Means and stdevs need to be dictionaries with keys as '
                    'feature names but received means of type '
                    f'{type(self._means)} and '
                    f'stdevs of type {type(self._stdevs)}'
                )
                logger.error(msg)
                raise TypeError(msg)

            missing = [f for f in self.lr_features if f not in self._means]
            missing += [
                f for f in self.hr_exo_features if f not in self._means
            ]
            missing += [
                f for f in self.hr_out_features if f not in self._means
            ]
            if any(missing):
                msg = (
                    f'Need means for features "{missing}" but did not find '
                    f'in new means array: {self._means}'
                )

            logger.info(
                'Set data normalization mean values:\n%s',
                pprint.pformat(self._means, indent=2),
            )
            logger.info(
                'Set data normalization stdev values:\n%s',
                pprint.pformat(self._stdevs, indent=2),
            )

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

            missing = [fn for fn in self.lr_features if fn not in self._means]
            if any(missing):
                msg = (
                    f'Could not find low-res input features {missing} in '
                    f'means/stdevs: {self._means}/{self._stdevs}'
                )
                logger.error(msg)
                raise KeyError(msg)

            means = np.array([self._means[fn] for fn in self.lr_features])
            stdevs = np.array([self._stdevs[fn] for fn in self.lr_features])
            if any(stdevs == 0):
                stdevs = np.where(stdevs == 0, 1, stdevs)
                msg = 'Some standard deviations are zero.'
                logger.warning(msg)
                warn(msg)
            low_res = (low_res.copy() - means) / stdevs

        return low_res

    def un_norm_output(self, output):
        """Un-normalize synthetically generated output data to physical units

        Parameters
        ----------
        output : tf.Tensor | np.ndarray
            Synthetically generated high-resolution data

        Returns
        -------
        output : np.ndarray
            Synthetically generated high-resolution data
        """
        if self._means is not None:
            if isinstance(output, tf.Tensor):
                output = output.numpy()

            missing = [
                fn for fn in self.hr_out_features if fn not in self._means
            ]
            if any(missing):
                msg = (
                    f'Could not find high-res output features {missing} in '
                    f'means/stdevs: {self._means}/{self._stdevs}'
                )
                logger.error(msg)
                raise KeyError(msg)

            means = [self._means[fn] for fn in self.hr_out_features]
            stdevs = [self._stdevs[fn] for fn in self.hr_out_features]
            means = np.array(means)
            stdevs = np.array(stdevs)

            output = (output * stdevs) + means

        return output

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
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        return self._history

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
            optimizer_class = getattr(optimizers, class_name)
            sig = signature(optimizer_class)
            optimizer_kwargs = {
                k: v for k, v in optimizer.items() if k in sig.parameters
            }
            optimizer = optimizer_class.from_config(optimizer_kwargs)
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
        with open(fp_params) as f:
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
                logger.info(
                    'Loading model from disk '
                    'that was created with the '
                    'following package versions: \n{}'.format(
                        pprint.pformat(version_record, indent=2)
                    )
                )

        means = params.get('means', None)
        stdevs = params.get('stdevs', None)
        if means is not None and stdevs is not None:
            means = {k: np.float32(v) for k, v in means.items()}
            stdevs = {k: np.float32(v) for k, v in stdevs.items()}
            params['means'] = means
            params['stdevs'] = stdevs

        return params

    def _init_records(self):
        """Initialize running records used to compute loss details running
        means"""
        if self._history is not None:
            train_cols = [c for c in self._history.columns if 'train_' in c]
            val_cols = [c for c in self._history.columns if 'val_' in c]
            self._train_record = self._history[train_cols].iloc[-1:]
            self._train_record = self._train_record.reset_index(drop=True)
            self._val_record = self._history[val_cols].iloc[-1:]
            self._val_record = self._val_record.reset_index(drop=True)

    def get_hr_exo_input(self, high_res):
        """Get exogenous feature data from high_res

        Parameters
        ----------
        high_res : tf.Tensor
            Ground truth high resolution spatiotemporal data.

        Returns
        -------
        exo_data : dict
            Dictionary of exogenous feature data used as input to tf_generate.
            e.g. ``{'topography': tf.Tensor(...)}``
        """
        exo_data = {}
        for feature in self.hr_exo_features:
            f_idx = self.hr_exo_features.index(feature)
            f_idx += len(self.hr_out_features)
            exo_fdata = high_res[..., f_idx : f_idx + 1]
            exo_data[feature] = exo_fdata
        return exo_data

    @tf.function
    def _combine_loss_input(self, high_res_true, high_res_gen):
        """Combine exogenous feature data from high_res_true with high_res_gen
        for loss calculation

        Parameters
        ----------
        high_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        high_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.

        Returns
        -------
        high_res_gen : tf.Tensor
            Same as input with exogenous data combined with high_res input
        """
        if high_res_true.shape[-1] > high_res_gen.shape[-1]:
            exo_dict = self.get_hr_exo_input(high_res_true)
            exo_data = [exo_dict[feat] for feat in self.hr_exo_features]
            high_res_gen = tf.concat((high_res_gen, *exo_data), axis=-1)
        return high_res_gen

    @classmethod
    def get_loss_fun(cls, loss):
        """Get full, possibly multi-term, loss function from the provided str
        or dictionary.

        Parameters
        ----------
        loss : str | dict
            Loss function class name from sup3r.utilities.loss_metrics
            (prioritized) or tensorflow.keras.losses or dictionary of loss
            function class names. As a dictionary this can include multiple
            loss function classes, each with dictionaries of kwargs for that
            function. Can also include a key ``term_weights``, which provides a
            list of weights for each loss function. e.g.
            ``{'SpatialExtremesLoss': {}, 'MeanAbsoluteError': {},
            'term_weights': [0.8, 0.2]}``

        Returns
        -------
        loss_func : tf.keras.losses.Loss
            Initialized loss function, possibly consisting of multiple
            individual functions. This loss function returns a total loss value
            and a dictionary of loss values for each loss term. For example, if
            the loss funcion is a weighted sum of ``MeanAbsoluteError`` and
            ``MeanSquaredError`` the dictionary will include entries for each
            of these functions.
        """
        loss = {loss: {}} if isinstance(loss, str) else loss
        lns = [ln for ln in loss if ln != 'term_weights']
        loss_funcs = [cls._get_loss_fun({ln: loss[ln]}) for ln in lns]
        weights = copy.deepcopy(loss).pop('term_weights', [1.0] * len(lns))

        def loss_fun(x1, x2):
            loss_details = {}
            loss = 0
            for i, (ln, loss_func) in enumerate(zip(lns, loss_funcs)):
                val = loss_func(x1, x2)
                loss_details[camel_to_underscore(ln)] = val
                loss += weights[i] * val
            return loss, loss_details

        return loss_fun

    @staticmethod
    def _get_loss_fun(loss):
        """Get the initialized loss function class from the sup3r loss library
        or the tensorflow losses.

        Parameters
        ----------
        loss : str | dict
            Loss function class name from sup3r.utilities.loss_metrics
            (prioritized) or tensorflow.keras.losses. Defaults to
            tf.keras.losses.MeanSquaredError. This can be provided as a dict
            with kwargs for loss functions with extra parameters.
            e.g. {'SpatialExtremesLoss': {'weight': 0.5}}

        Returns
        -------
        out : tf.keras.losses.Loss
            Initialized loss function class that is callable, e.g. if
            "MeanSquaredError" is requested, this will return
            an instance of tf.keras.losses.MeanSquaredError()
        """
        kwargs = {}
        if isinstance(loss, dict):
            loss, kwargs = next(iter(loss.items()))

        out = getattr(sup3r.utilities.loss_metrics, loss, None)
        if out is None:
            out = getattr(tf.keras.losses, loss, None)

        if out is None:
            msg = (
                'Could not find requested loss function "{}" in '
                'sup3r.utilities.loss_metrics or tf.keras.losses.'.format(loss)
            )
            logger.error(msg)
            raise KeyError(msg)

        return out(**kwargs)

    @staticmethod
    def get_optimizer_config(optimizer):
        """Get a config that defines the current model optimizer

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
            TF-Keras optimizer object (e.g., Adam)

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

    @classmethod
    def get_optimizer_state(cls, optimizer):
        """Get a set of state variables for the optimizer

        Parameters
        ----------
        optimizer : tf.keras.optimizers.Optimizer
            TF-Keras optimizer object (e.g., Adam)

        Returns
        -------
        state : dict
            Optimizer state variables
        """
        lr = cls.get_optimizer_config(optimizer)['learning_rate']
        state = {'learning_rate': lr}
        for var in optimizer.variables:
            name = var.name
            var = var.numpy().flatten()
            var = np.abs(var).mean()  # collapse ndarrays into mean absolute
            state[name] = float(var)
        return state

    @staticmethod
    def update_loss_details(record, new_data, max_batches, prefix=None):
        """Update a dictionary of loss_details with loss information from a new
        batch.

        Parameters
        ----------
        record : pd.DataFrame
            Details for the last N batches, where N is the number of batches in
            an epoch, used to compute the running means.
        new_data : dict
            Namespace of the breakdown of loss components for a single new
            batch.
        max_batches : int
            Maximum number of batches to use for the running mean of loss
            details
        prefix : None | str
            Option to prefix the names of the loss data when saving to the
            loss_details dictionary. This is usually 'train_' or 'val_'

        Returns
        -------
        record : pd.DataFrame
            Same as input with details from ``new_data`` added and only the
            last ``max_batches`` rows kept.
        """
        new_index = 0 if len(record) == 0 else record.index[-1] + 1
        for k, v in new_data.items():
            # only add prefix if key doesn't already include the prefix - no
            # point in adding 'train_' to keys like 'disc_train_frac'
            key = k if prefix is None or prefix in k else prefix + k
            new_value = numpy_if_tensor(v)
            record.loc[new_index, key] = new_value
        return record.iloc[-max_batches:]

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
            msg_format = '\t{}: {}' if isinstance(v, str) else '\t{}: {:.2e}'
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
                logger.info(
                    'Found early stop condition, loss values "{}" '
                    'have absolute relative differences less than '
                    'threshold {}: {}'.format(
                        column, threshold, diffs[-n_epoch:]
                    )
                )

        return stop

    @abstractmethod
    def save(self, out_dir):
        """Save the model with its sub-networks to a directory.

        Parameters
        ----------
        out_dir : str
            Directory to save model files. This directory will be created
            if it does not already exist.
        """

    def finish_epoch(
        self,
        epoch,
        epochs,
        t0,
        loss_details,
        checkpoint_int,
        out_dir,
        early_stop_on,
        early_stop_threshold,
        early_stop_n_epoch,
        extras=None,
    ):
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
            Directory to save checkpoint models. Should have {epoch} in
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
        entry = np.vstack(list(loss_details.values())).T
        self._history.loc[epoch, list(loss_details.keys())] = entry

        last_epoch = epoch == epochs[-1]
        chp = checkpoint_int is not None and (epoch % checkpoint_int) == 0
        if last_epoch or chp:
            msg = (
                'Model output dir for checkpoint models should have '
                f'{"{epoch}"} but did not: {out_dir}'
            )
            assert '{epoch}' in out_dir, msg
            self.save(out_dir.format(epoch=epoch))

        stop = False
        if early_stop_on is not None and early_stop_on in self._history:
            stop = self.early_stop(
                self._history,
                early_stop_on,
                threshold=early_stop_threshold,
                n_epoch=early_stop_n_epoch,
            )
            if stop:
                self.save(out_dir.format(epoch=epoch))

        if extras is not None:
            for k, v in extras.items():
                self._history.at[epoch, k] = safe_cast(v)

        return stop

    def run_gradient_descent(
        self,
        low_res,
        hi_res_true,
        training_weights,
        optimizer=None,
        multi_gpu=False,
        **calc_loss_kwargs,
    ):
        # pylint: disable=E0602
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
            futures = []
            lr_chunks = np.array_split(low_res, len(self.gpu_list))
            hr_true_chunks = np.array_split(hi_res_true, len(self.gpu_list))
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
                            device_name=f'/gpu:{i}',
                            **calc_loss_kwargs,
                        )
                    )

            # sum the gradients from each gpu to weight equally in
            # optimizer momentum calculation
            total_grad = None
            for future in futures:
                grad, loss_details = future.result()
                if total_grad is None:
                    total_grad = grad
                else:
                    for i, igrad in enumerate(grad):
                        total_grad[i] += igrad

            optimizer.apply_gradients(zip(total_grad, training_weights))

            self.timer.stop()
            logger.debug(
                'Finished %s gradient descent steps on %s GPUs in %s',
                len(futures),
                len(self.gpu_list),
                self.timer.elapsed_str,
            )
        return loss_details

    def _reshape_norm_exo(self, hi_res, hi_res_exo, exo_name, norm_in=True):
        """Reshape the hi_res_topo to match the hi_res tensor (if necessary)
        and normalize (if requested).

        Parameters
        ----------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        hi_res_exo : np.ndarray
            This should be a 4D array for spatial enhancement model or 5D array
            for a spatiotemporal enhancement model (obs, spatial_1, spatial_2,
            (temporal), features) corresponding to the high-resolution
            spatial_1, spatial_2, temporal. This data will be input to the
            custom phygnn Sup3rAdder or Sup3rConcat layer if found in the
            generative network. This differs from the exogenous_data input in
            that exogenous_data always matches the low-res input. For this
            function, hi_res_exo can also be a 3D array (spatial_1, spatial_2,
            1). Note that this input gets normalized if norm_in=True.
        exo_name : str
            Name of feature corresponding to hi_res_exo data.
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The generator should always
            received normalized data with mean=0 stdev=1. This also normalizes
            hi_res_topo.

        Returns
        -------
        hi_res_topo : np.ndarray
            Same as input but reshaped to match hi_res (if necessary) and
            normalized (if requested)
        """
        if hi_res_exo is None:
            return hi_res_exo

        if norm_in and self._means is not None:
            hi_res_exo = (
                hi_res_exo.copy() - self._means[exo_name]
            ) / self._stdevs[exo_name]

        if len(hi_res_exo.shape) == 3:
            hi_res_exo = np.expand_dims(hi_res_exo, axis=0)
            hi_res_exo = np.repeat(hi_res_exo, hi_res.shape[0], axis=0)
        if len(hi_res_exo.shape) == 4 and len(hi_res.shape) == 5:
            hi_res_exo = np.expand_dims(hi_res_exo, axis=3)
            hi_res_exo = np.repeat(hi_res_exo, hi_res.shape[3], axis=3)

        if len(hi_res_exo.shape) != len(hi_res.shape):
            msg = (
                'hi_res and hi_res_exo arrays are not of the same rank: '
                '{} and {}'.format(hi_res.shape, hi_res_exo.shape)
            )
            logger.error(msg)
            raise RuntimeError(msg)

        return hi_res_exo

    def generate(
        self, low_res, norm_in=True, un_norm_out=True, exogenous_data=None
    ):
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
        exogenous_data : dict | ExoData | None
            Special dictionary (class:`ExoData`) of exogenous feature data with
            entries describing whether features should be combined at input, a
            mid network layer, or with output. This doesn't have to include
            the 'model' key since this data is for a single step model.

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        if (
            not isinstance(exogenous_data, ExoData)
            and exogenous_data is not None
        ):
            exogenous_data = ExoData(exogenous_data)

        low_res = self._combine_fwp_input(low_res, exogenous_data)
        if norm_in and self._means is not None:
            low_res = self.norm_input(low_res)

        hi_res = self.generator.layers[0](low_res)
        layer_num = 1
        try:
            for i, layer in enumerate(self.generator.layers[1:]):
                layer_num = i + 1
                if isinstance(layer, (Sup3rAdder, Sup3rConcat)):
                    msg = (
                        f'layer.name = {layer.name} does not match any '
                        'features in exogenous_data '
                        f'({list(exogenous_data)})'
                    )
                    assert layer.name in exogenous_data, msg
                    hi_res_exo = exogenous_data.get_combine_type_data(
                        layer.name, 'layer'
                    )
                    hi_res_exo = self._reshape_norm_exo(
                        hi_res, hi_res_exo, layer.name, norm_in=norm_in
                    )
                    hi_res = layer(hi_res, hi_res_exo)
                else:
                    hi_res = layer(hi_res)
        except Exception as e:
            msg = 'Could not run layer #{} "{}" on tensor of shape {}'.format(
                layer_num, layer, hi_res.shape
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

        hi_res = hi_res.numpy()

        if un_norm_out and self._means is not None:
            hi_res = self.un_norm_output(hi_res)

        return self._combine_fwp_output(hi_res, exogenous_data)

    @tf.function
    def _tf_generate(self, low_res, hi_res_exo=None):
        """Use the generator model to generate high res data from low res input

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data. The generator should always
            received normalized data with mean=0 stdev=1.
        hi_res_exo : dict
            Dictionary of exogenous_data with same resolution as high_res data
            e.g. ``{'topography': np.array}``
            The arrays in this dictionary should be a 4D array for spatial
            enhancement model or 5D array for a spatiotemporal enhancement
            model ``(obs, spatial_1, spatial_2, (temporal), features)``
            corresponding to the high-resolution spatial_1 and spatial_2. This
            data will be input to the custom phygnn Sup3rAdder or Sup3rConcat
            layer if found in the generative network. This differs from the
            exogenous_data input in that exogenous_data always matches the
            low-res input.

        Returns
        -------
        hi_res : tf.Tensor
            Synthetically generated high-resolution data
        """
        hi_res = self.generator.layers[0](low_res)
        layer_num = 1
        try:
            for i, layer in enumerate(self.generator.layers[1:]):
                layer_num = i + 1
                if isinstance(layer, (Sup3rAdder, Sup3rConcat)):
                    msg = (
                        f'layer.name = {layer.name} does not match any '
                        f'features in exogenous_data ({list(hi_res_exo)})'
                    )
                    assert layer.name in hi_res_exo, msg
                    hr_exo = hi_res_exo[layer.name]
                    hi_res = layer(hi_res, hr_exo)
                else:
                    hi_res = layer(hi_res)
        except Exception as e:
            msg = 'Could not run layer #{} "{}" on tensor of shape {}'.format(
                layer_num, layer, hi_res.shape
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

        return hi_res

    @tf.function
    def get_single_grad(
        self,
        low_res,
        hi_res_true,
        training_weights,
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
            hi_res_exo = self.get_hr_exo_input(hi_res_true)
            hi_res_gen = self._tf_generate(low_res, hi_res_exo)
            loss_out = self.calc_loss(
                hi_res_true, hi_res_gen, **calc_loss_kwargs
            )
            loss, loss_details = loss_out
            grad = tape.gradient(loss, training_weights)
        return grad, loss_details
