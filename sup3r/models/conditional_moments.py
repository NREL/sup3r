# -*- coding: utf-8 -*-
"""Sup3r model software"""
import os
import time
import logging
import numpy as np
import pprint
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from warnings import warn
from rex.utilities.utilities import safe_json_load
from phygnn import CustomNetwork

from sup3r.models.abstract import AbstractInterface, AbstractSingleModel
from sup3r.utilities import VERSION_RECORD


logger = logging.getLogger(__name__)


class Sup3rCondMom(AbstractInterface, AbstractSingleModel):
    """Basic Sup3r conditional moments model."""

    def __init__(self, gen_layers,
                 optimizer=None, learning_rate=1e-4, num_par=None,
                 history=None, meta=None, means=None, stdevs=None, name=None):
        """
        Parameters
        ----------
        gen_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            generative super resolving model. Can also be a str filepath to a
            .json config file containing the input layers argument or a .pkl
            for a saved pre-trained model.
        optimizer : tf.keras.optimizers.Optimizer | dict | None | str
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
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
            Optional name for the model.
        """

        self._version_record = VERSION_RECORD
        self.name = name if name is not None else self.__class__.__name__
        self._meta = meta if meta is not None else {}
        self._num_par = num_par if num_par is not None else 0
        self.loss_name = 'MeanSquaredError'
        self.loss_fun = self.get_loss_fun(self.loss_name)

        self._history = history
        if isinstance(self._history, str):
            self._history = pd.read_csv(self._history, index_col=0)

        self._optimizer = self.init_optimizer(optimizer, learning_rate)

        self._gen = self.load_network(gen_layers, 'generator')

        self._means = (means if means is None
                       else np.array(means).astype(np.float32))
        self._stdevs = (stdevs if stdevs is None
                        else np.array(stdevs).astype(np.float32))

    def save(self, out_dir):
        """Save the model with its sub-networks to a directory.

        Parameters
        ----------
        out_dir : str
            Directory to save model files. This directory will be created
            if it does not already exist.
        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        fp_gen = os.path.join(out_dir, 'model_gen.pkl')
        self.generator.save(fp_gen)

        fp_history = None
        if isinstance(self.history, pd.DataFrame):
            fp_history = os.path.join(out_dir, 'history.csv')
            self.history.to_csv(fp_history)

        self.save_params(out_dir)

        logger.info('Saved model to disk in directory: {}'.format(out_dir))

    @classmethod
    def load(cls, model_dir, verbose=True):
        """Load the model with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        model_dir : str
            Directory to load model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : BaseModel
            Returns a pretrained gan model that was previously saved to out_dir
        """
        if verbose:
            logger.info('Loading model from disk in directory: {}'
                        .format(model_dir))
            msg = ('Active python environment versions: \n{}'
                   .format(pprint.pformat(VERSION_RECORD, indent=4)))
            logger.info(msg)

        fp_gen = os.path.join(model_dir, 'model_gen.pkl')
        params = cls.load_saved_params(model_dir, verbose=verbose)
        return cls(fp_gen, **params)

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

            for idf in range(output.shape[-1]):
                feature_name = self.output_features[idf]
                i = self.training_features.index(feature_name)
                mean = self._means[i]
                stdev = self._stdevs[i]
                output[..., idf] = (output[..., idf] * stdev) + mean

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
        output : ndarray
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

        output = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
                output = layer(output)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, output.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        output = output.numpy()

        if un_norm_out and self._means is not None:
            output = self.un_norm_output(output)

        return output

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
        output : tf.Tensor
            Synthetically generated high-resolution data
        """

        output = self.generator.layers[0](low_res)
        for i, layer in enumerate(self.generator.layers[1:]):
            try:
                output = layer(output)
            except Exception as e:
                msg = ('Could not run layer #{} "{}" on tensor of shape {}'
                       .format(i + 1, layer, output.shape))
                logger.error(msg)
                raise RuntimeError(msg) from e

        return output

    def update_optimizer(self, **kwargs):
        """Update optimizer by changing current configuration

        Parameters
        ----------
        kwargs : dict
            kwargs to use for optimizer configuration update
        """

        conf = self.get_optimizer_config(self.optimizer)
        conf.update(**kwargs)
        OptimizerClass = getattr(optimizers, conf['name'])
        self._optimizer = OptimizerClass.from_config(conf)

    @property
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

        if 'class' not in self._meta:
            self._meta['class'] = self.__class__.__name__

        return self._meta

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

        num_par = int(np.sum(
                      [np.prod(v.get_shape().as_list())
                       for v in self.weights]))

        model_params = {'name': self.name,
                        'num_par': num_par,
                        'version_record': self.version_record,
                        'optimizer': config_optm_g,
                        'means': means,
                        'stdevs': stdevs,
                        'meta': self.meta,
                        }

        return model_params

    @property
    def weights(self):
        """Get a list of all the layer weights and bias terms for the
        generator network
        """
        return self.generator_weights

    def run_gradient_descent(self, low_res, output_true, mask,
                             training_weights,
                             optimizer=None, **calc_loss_kwargs):
        """Run gradient descent for one mini-batch of (low_res, output_true)
        and adjust NN weights

        Parameters
        ----------
        low_res : np.ndarray
            Real low-resolution data in a 4D or 5D array:
            (n_observations, spatial_1, spatial_2, features)
            (n_observations, spatial_1, spatial_2, temporal, features)
        output_true : np.ndarray
            Real high-resolution data in a 4D or 5D array:
            (n_observations, spatial_1, spatial_2, features)
            (n_observations, spatial_1, spatial_2, temporal, features)
        mask : np.ndarray
            Mask of high-resolution data in a 4D or 5D array:
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

            output_gen = self._tf_generate(low_res)
            loss_out = self.calc_loss(output_true, output_gen, mask,
                                      **calc_loss_kwargs)
            loss, loss_details = loss_out

            grad = tape.gradient(loss, training_weights)

        if optimizer is None:
            optimizer = self.optimizer

        optimizer.apply_gradients(zip(grad, training_weights))

        return loss_details

    @tf.function
    def calc_loss_cond_mom(self, output_true, output_gen, mask):
        """Calculate the loss of the moment predictor

        Parameters
        ----------
        output_true : tf.Tensor
            True realization output
        output_gen : tf.Tensor
            Predicted realization output
        mask : tf.Tensor
            Mask to apply

        Returns
        -------
        loss : tf.Tensor
            0D tensor generator model loss for the MSE loss of the
            moment predictor
        """

        loss = self.loss_fun(output_true * mask,
                             output_gen * mask)

        return loss

    def calc_loss(self, output_true, output_gen, mask):
        """Calculate the total moment predictor loss

        Parameters
        ----------
        output_true : tf.Tensor
            True realization output
        output_gen : tf.Tensor
            Predicted realization output
        mask : tf.Tensor
            Mask to apply

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the
            moment predictor
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        if output_gen.shape != output_true.shape:
            msg = ('The tensor shapes of the synthetic output {} and '
                   'true output {} did not have matching shape! '
                   'Check the spatiotemporal enhancement multipliers in your '
                   'your model config and data handlers.'
                   .format(output_gen.shape, output_true.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        loss = self.calc_loss_cond_mom(output_true, output_gen, mask)

        # print("min max out = %g / %g " %
        #       (np.amin(output_gen), np.amax(output_gen)))
        # print("min max true = %g / %g " %
        #       (np.amin(output_true), np.amax(output_true)))

        loss_details = {'loss_gen': loss}

        return loss, loss_details

    def calc_val_loss(self, batch_handler, loss_details):
        """Calculate the validation loss at the current state of model training

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
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
            output_gen = self._tf_generate(val_batch.low_res)
            _, v_loss_details = self.calc_loss(
                val_batch.output, output_gen, val_batch.mask)

            loss_details = self.update_loss_details(loss_details,
                                                    v_loss_details,
                                                    len(val_batch),
                                                    prefix='val_')

        return loss_details

    def train_epoch(self, batch_handler):
        """Train the model for one epoch.

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        loss_details = {'n_obs': 0}

        for ib, batch in enumerate(batch_handler):
            b_loss_details = {}
            b_loss_details = self.run_gradient_descent(
                batch.low_res, batch.output, batch.mask,
                self.generator_weights,
                optimizer=self.optimizer)

            loss_details = self.update_loss_details(loss_details,
                                                    b_loss_details,
                                                    len(batch),
                                                    prefix='train_')

            logger.debug('Batch {} out of {} has epoch-average '
                         'gen loss of: {:.2e}. '
                         .format(ib, len(batch_handler),
                                 loss_details['train_loss_gen']))

        return loss_details

    def train(self, batch_handler, n_epoch,
              checkpoint_int=None,
              out_dir='./condMom_{epoch}',
              early_stop_on=None,
              early_stop_threshold=0.005,
              early_stop_n_epoch=5):
        """Train the model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        n_epoch : int
            Number of epochs to train on
        checkpoint_int : int | None
            Epoch interval at which to save checkpoint models.
        out_dir : str
            Directory to save checkpoint models. Should have {epoch} in
            the directory name. This directory will be created if it does not
            already exist.
        early_stop_on : str | None
            If not None, this should be a column in the training history to
            evaluate for early stopping (e.g. validation_loss_gen).
            If this value in this history decreases by
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
        logger.info('Training model '
                    'for {} epochs starting at epoch {}'
                    .format(n_epoch, epochs[0]))

        for epoch in epochs:
            loss_details = self.train_epoch(batch_handler)

            loss_details = self.calc_val_loss(batch_handler, loss_details)

            msg = f'Epoch {epoch} of {epochs[-1]} '
            msg += 'gen train loss: {:.2e} '.format(
                loss_details["train_loss_gen"])

            if all(loss in loss_details for loss
                   in ['val_loss_gen']):
                msg += 'gen val loss: {:.2e} '.format(
                    loss_details["val_loss_gen"])

            logger.info(msg)

            lr_g = self.get_optimizer_config(self.optimizer)['learning_rate']

            extras = {'learning_rate_gen': lr_g}

            stop = self.finish_epoch(epoch, epochs, t0, loss_details,
                                     checkpoint_int, out_dir,
                                     early_stop_on, early_stop_threshold,
                                     early_stop_n_epoch, extras=extras)

            if stop:
                break

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
        for key, value in loss_details.items():
            if key != 'n_obs':
                self._history.at[epoch, key] = value

        last_epoch = epoch == epochs[-1]
        chp = checkpoint_int is not None and (epoch % checkpoint_int) == 0
        if last_epoch or chp:
            msg = ('Model output dir for checkpoint models should have '
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
