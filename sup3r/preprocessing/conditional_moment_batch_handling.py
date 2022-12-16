# -*- coding: utf-8 -*-
"""
Sup3r batch_handling module.
"""
import logging
import numpy as np
from datetime import datetime as dt

from rex.utilities import log_mem

from sup3r.utilities.utilities import (spatial_coarsening,
                                       temporal_coarsening,
                                       spatial_simple_enhancing,
                                       temporal_simple_enhancing,
                                       smooth_data)
from sup3r.preprocessing.batch_handling import (Batch,
                                                ValidationData,
                                                BatchHandler)

np.random.seed(42)

logger = logging.getLogger(__name__)


class BatchMom1(Batch):
    """Batch of low_res, high_res and output data"""

    def __init__(self, low_res, high_res, output, mask):
        """Stores low, high res, output and mask data

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        output : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        mask : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """
        self._low_res = low_res
        self._high_res = high_res
        self._output = output
        self._mask = mask

    @property
    def output(self):
        """Get the output for the batch.
           Output predicted by the neural net can be different
           than the high_res when doing moment estimation.
           For ex: output may be (high_res)**2
           We distinguish output from high_res since it may not be
           possible to recover high_res from output."""
        return self._output

    @property
    def mask(self):
        """Get the mask for the batch."""
        return self._mask

    # pylint: disable=W0613
    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None):
        """Make custom batch output

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int | None
            Spatial enhancement factor
        t_enhance : int | None
            Temporal enhancement factor
        model_mom1 : Sup3rCondMom | None
            Model used to modify the make the batch output
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        return high_res

    # pylint: disable=E1130
    @staticmethod
    def make_mask(high_res, s_padding=None, t_padding=None):
        """Make mask for output.
        The mask is used to ensure consistency when training conditional
        moments.
        Consider the case of learning E(HR|LR) where HR is the high_res and
        LR is the low_res.
        In theory, the conditional moment estimation works if
        the full LR is passed as input and predicts the full HR.
        In practice, only the LR data that overlaps and surrounds the HR data
        is useful, ie E(HR|LR) = E(HR|LR_nei) where LR_nei is the LR data
        that surrounds the HR data. Physically, this is equivalent to saying
        that data far away from a region of interest does not matter.
        This allows learning the conditional moments on spatial and
        temporal chunks only if one restricts the high_res output as being
        overlapped and surrounded by the input low_res.
        The role of the mask is to ensure that the input low_res always
        surrounds the output high_res.

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_padding : int | None
            Spatial padding size. If None or 0, no padding is applied.
            None by default
        t_padding : int | None
            Temporal padding size. If None or 0, no padding is applied.
            None by default
        model_mom1 : Sup3rCondMom | None
            Model used to modify the make the batch output
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        mask = np.zeros(high_res.shape, dtype=np.float32)
        s_min = s_padding if s_padding is not None else 0
        t_min = t_padding if t_padding is not None else 0
        s_max = -s_padding if s_min > 0 else None
        t_max = -t_padding if t_min > 0 else None

        if len(high_res.shape) == 4:
            mask[:, s_min:s_max, s_min:s_max, :] = 1.0
        elif len(high_res.shape) == 5:
            mask[:, s_min:s_max, s_min:s_max, t_min:t_max, :] = 1.0

        return mask

    # pylint: disable=W0613
    @classmethod
    def get_coarse_batch(cls, high_res,
                         s_enhance, t_enhance=1,
                         temporal_coarsening_method='subsample',
                         output_features_ind=None,
                         output_features=None,
                         training_features=None,
                         smoothing=None,
                         smoothing_ignore=None,
                         model_mom1=None,
                         s_padding=None,
                         t_padding=None):
        """Coarsen high res data and return Batch with high res and
        low res data

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data
        temporal_coarsening_method : str
            Method to use for temporal coarsening. Can be subsample, average,
            or total
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names
        training_features : list | None
            Ordered list of training features input to the generative model
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None
        model_mom1 : Sup3rCondMom | None
            Model used to modify the make the batch output
        s_padding : int | None
            Width of spatial padding to predict only middle part. If None,
            no padding is used
        t_padding : int | None
            Width of temporal padding to predict only middle part. If None,
            no padding is used

        Returns
        -------
        Batch
            Batch instance with low and high res data
        """
        low_res = spatial_coarsening(high_res, s_enhance)

        if training_features is None:
            training_features = [None] * low_res.shape[-1]

        if smoothing_ignore is None:
            smoothing_ignore = []

        if t_enhance != 1:
            low_res = temporal_coarsening(low_res, t_enhance,
                                          temporal_coarsening_method)

        low_res = smooth_data(low_res, training_features, smoothing_ignore,
                              smoothing)
        high_res = cls.reduce_features(high_res, output_features_ind)
        output = cls.make_output(low_res, high_res,
                                 s_enhance, t_enhance,
                                 model_mom1, output_features_ind)
        mask = cls.make_mask(high_res,
                             s_padding, t_padding)
        batch = cls(low_res, high_res, output, mask)

        return batch


class BatchMom1SF(BatchMom1):
    """Batch of low_res, high_res and output data
    when learning first moment of subfilter vel"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None):
        """Make custom batch output

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int | None
            Spatial enhancement factor
        t_enhance : int | None
            Temporal enhancement factor
        model_mom1 : Sup3rCondMom | None
            Model used to modify the make the batch output
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        # Remove LR from HR
        enhanced_lr = spatial_simple_enhancing(low_res,
                                               s_enhance=s_enhance)
        enhanced_lr = temporal_simple_enhancing(enhanced_lr,
                                                t_enhance=t_enhance)
        enhanced_lr = Batch.reduce_features(enhanced_lr, output_features_ind)
        return high_res - enhanced_lr


class BatchMom2(BatchMom1):
    """Batch of low_res, high_res and output data
    when learning second moment"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None):
        """Make custom batch output

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int | None
            Spatial enhancement factor
        t_enhance : int | None
            Temporal enhancement factor
        model_mom1 : Sup3rCondMom | None
            Model used to modify the make the batch output
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        # Remove first moment from HR and square it
        out = model_mom1._tf_generate(low_res).numpy()
        return (high_res - out)**2


class BatchMom2Sep(BatchMom1):
    """Batch of low_res, high_res and output data
    when learning second moment separate from first moment"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None):
        """Make custom batch output

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int | None
            Spatial enhancement factor
        t_enhance : int | None
            Temporal enhancement factor
        model_mom1 : Sup3rCondMom | None
            Model used to modify the make the batch output
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        return super(BatchMom2Sep,
                     BatchMom2Sep).make_output(low_res, high_res,
                                               s_enhance, t_enhance,
                                               model_mom1,
                                               output_features_ind)**2


class BatchMom2SF(BatchMom1):
    """Batch of low_res, high_res and output data
    when learning second moment of subfilter vel"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None):
        """Make custom batch output

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int | None
            Spatial enhancement factor
        t_enhance : int | None
            Temporal enhancement factor
        model_mom1 : Sup3rCondMom | None
            Model used to modify the make the batch output
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        # Remove LR and first moment from HR and square it
        out = model_mom1._tf_generate(low_res).numpy()
        enhanced_lr = spatial_simple_enhancing(low_res,
                                               s_enhance=s_enhance)
        enhanced_lr = temporal_simple_enhancing(enhanced_lr,
                                                t_enhance=t_enhance)
        enhanced_lr = Batch.reduce_features(enhanced_lr, output_features_ind)
        return (high_res - enhanced_lr - out)**2


class BatchMom2SepSF(BatchMom1SF):
    """Batch of low_res, high_res and output data
    when learning second moment of subfilter vel
    separate from first moment"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None):
        """Make custom batch output

        Parameters
        ----------
        low_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        s_enhance : int | None
            Spatial enhancement factor
        t_enhance : int | None
            Temporal enhancement factor
        model_mom1 : Sup3rCondMom | None
            Model used to modify the make the batch output
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        """
        # Remove LR from HR and square it
        return super(BatchMom2SepSF,
                     BatchMom2SepSF).make_output(low_res, high_res,
                                                 s_enhance, t_enhance,
                                                 model_mom1,
                                                 output_features_ind)**2


class ValidationDataMom1(ValidationData):
    """Iterator for validation data"""

    # Classes to use for handling an individual batch obj.
    BATCH_CLASS = BatchMom1

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=1,
                 temporal_coarsening_method='subsample',
                 output_features_ind=None,
                 output_features=None,
                 smoothing=None, smoothing_ignore=None,
                 model_mom1=None,
                 s_padding=None, t_padding=None):
        """
        Parameters
        ----------
        handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Size of validation data batches
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        output_features_ind : list | np.ndarray | None
            List/array of feature channel indices that are used for generative
            output, without any feature indices used only for training.
        output_features : list
            List of Generative model output feature names
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None
        model_mom1 : Sup3rCondMom | None
            model that predicts the first conditional moments.
            Useful to prepare data for learning second conditional moment.
        s_padding : int | None
            Width of spatial padding to predict only middle part. If None,
            no padding is used
        t_padding : int | None
            Width of temporal padding to predict only middle part. If None,
            no padding is used
        """

        handler_shapes = np.array([d.sample_shape for d in data_handlers])
        assert np.all(handler_shapes[0] == handler_shapes)

        self.handlers = data_handlers
        self.batch_size = batch_size
        self.sample_shape = handler_shapes[0]
        self.val_indices = self._get_val_indices()
        self.max = np.ceil(
            len(self.val_indices) / (batch_size))
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.s_padding = s_padding
        self.t_padding = t_padding
        self._remaining_observations = len(self.val_indices)
        self.temporal_coarsening_method = temporal_coarsening_method
        self._i = 0
        self.output_features_ind = output_features_ind
        self.output_features = output_features
        self.smoothing = smoothing
        self.smoothing_ignore = smoothing_ignore
        self.model_mom1 = model_mom1

    def batch_next(self, high_res):
        """Assemble the next batch

        Parameters
        ----------
        high_res : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        batch : Batch
        """
        return self.BATCH_CLASS.get_coarse_batch(
            high_res, self.s_enhance,
            t_enhance=self.t_enhance,
            temporal_coarsening_method=self.temporal_coarsening_method,
            output_features_ind=self.output_features_ind,
            smoothing=self.smoothing,
            smoothing_ignore=self.smoothing_ignore,
            output_features=self.output_features,
            model_mom1=self.model_mom1,
            s_padding=self.s_padding,
            t_padding=self.t_padding)


class BatchHandlerMom1(BatchHandler):
    """Sup3r base batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationDataMom1
    BATCH_CLASS = BatchMom1
    DATA_HANDLER_CLASS = None

    def __init__(self, data_handlers, batch_size=8, s_enhance=3, t_enhance=1,
                 means=None, stds=None, norm=True, n_batches=10,
                 temporal_coarsening_method='subsample', stdevs_file=None,
                 means_file=None, overwrite_stats=False, smoothing=None,
                 smoothing_ignore=None, stats_workers=None, norm_workers=None,
                 load_workers=None, max_workers=None, model_mom1=None,
                 s_padding=None, t_padding=None):
        """
        Parameters
        ----------
        data_handlers : list[DataHandler]
            List of DataHandler instances
        batch_size : int
            Number of observations in a batch
        s_enhance : int
            Factor by which to coarsen spatial dimensions of the high
            resolution data to generate low res data
        t_enhance : int
            Factor by which to coarsen temporal dimension of the high
            resolution data to generate low res data
        means : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data
            features. If not None and norm is True these will be used for
            normalization
        stds : np.ndarray
            dimensions (features)
            array of means for all features with same ordering as data
            features.  If not None and norm is True these will be used form
            normalization
        norm : bool
            Whether to normalize the data or not
        n_batches : int
            Number of batches in an epoch, this sets the iteration limit for
            this object.
        temporal_coarsening_method : str
            [subsample, average, total]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps, total will sum over t_enhance
            time steps
        stdevs_file : str | None
            Path to stdevs data or where to save data after calling get_stats
        means_file : str | None
            Path to means data or where to save data after calling get_stats
        overwrite_stats : bool
            Whether to overwrite stats cache files.
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None
        max_workers : int | None
            Providing a value for max workers will be used to set the value of
            norm_workers, stats_workers, and load_workers.
            If max_workers == 1 then all processes will be serialized. If None
            stats_workers, load_workers, and norm_workers will use their own
            provided values.
        load_workers : int | None
            max number of workers to use for loading data handlers.
        norm_workers : int | None
            max number of workers to use for normalizing data handlers.
        stats_workers : int | None
            max number of workers to use for computing stats across data
            handlers.
        model_mom1 : Sup3rCondMom | None
            model that predicts the first conditional moments.
            Useful to prepare data for learning second conditional moment.
        s_padding : int | None
            Width of spatial padding to predict only middle part. If None,
            no padding is used
        t_padding : int | None
            Width of temporal padding to predict only middle part. If None,
            no padding is used
        """
        if max_workers is not None:
            norm_workers = stats_workers = load_workers = max_workers

        msg = ('All data handlers must have the same sample_shape')
        handler_shapes = np.array([d.sample_shape for d in data_handlers])
        assert np.all(handler_shapes[0] == handler_shapes), msg

        self.data_handlers = data_handlers
        self._i = 0
        self.low_res = None
        self.high_res = None
        self.output = None
        self.batch_size = batch_size
        self._val_data = None
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.s_padding = s_padding
        self.t_padding = t_padding
        self.sample_shape = handler_shapes[0]
        self.means = means
        self.stds = stds
        self.n_batches = n_batches
        self.temporal_coarsening_method = temporal_coarsening_method
        self.current_batch_indices = None
        self.current_handler_index = None
        self.stdevs_file = stdevs_file
        self.means_file = means_file
        self.overwrite_stats = overwrite_stats
        self.smoothing = smoothing
        self.smoothing_ignore = smoothing_ignore or []
        self.smoothed_features = [f for f in self.training_features
                                  if f not in self.smoothing_ignore]
        self._stats_workers = stats_workers
        self._norm_workers = norm_workers
        self._load_workers = load_workers
        self.model_mom1 = model_mom1

        logger.info(f'Initializing BatchHandler with smoothing={smoothing}. '
                    f'Using stats_workers={self.stats_workers}, '
                    f'norm_workers={self.norm_workers}, '
                    f'load_workers={self.load_workers}.')

        now = dt.now()
        self.parallel_load()
        logger.debug(f'Finished loading data of shape {self.shape} '
                     f'for BatchHandler in {dt.now() - now}.')
        log_mem(logger, log_level='INFO')

        if norm:
            self.means, self.stds = self.check_cached_stats()
            self.normalize(self.means, self.stds)

        logger.debug('Getting validation data for BatchHandler.')
        self.val_data = self.VAL_CLASS(
            data_handlers, batch_size=batch_size,
            s_enhance=s_enhance, t_enhance=t_enhance,
            temporal_coarsening_method=temporal_coarsening_method,
            output_features_ind=self.output_features_ind,
            output_features=self.output_features,
            smoothing=self.smoothing,
            smoothing_ignore=self.smoothing_ignore,
            model_mom1=self.model_mom1,
            s_padding=self.s_padding,
            t_padding=self.t_padding)

        logger.info('Finished initializing BatchHandler.')
        log_mem(logger, log_level='INFO')

    def __next__(self):
        """Get the next iterator output.

        Returns
        -------
        batch : Batch
            Batch object with batch.low_res, batch.high_res
            and batch.output attributes with the appropriate coarsening.
        """
        self.current_batch_indices = []
        if self._i < self.n_batches:
            handler_index = np.random.randint(0, len(self.data_handlers))
            self.current_handler_index = handler_index
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.sample_shape[2],
                                 self.shape[-1]), dtype=np.float32)

            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next()
                self.current_batch_indices.append(handler.current_obs_index)

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance, t_enhance=self.t_enhance,
                temporal_coarsening_method=self.temporal_coarsening_method,
                output_features_ind=self.output_features_ind,
                output_features=self.output_features,
                training_features=self.training_features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore,
                model_mom1=self.model_mom1,
                s_padding=self.s_padding,
                t_padding=self.t_padding)

            self._i += 1
            return batch
        else:
            raise StopIteration


class SpatialBatchHandlerMom1(BatchHandlerMom1):
    """Sup3r spatial batch handling class"""

    def __next__(self):
        if self._i < self.n_batches:
            handler_index = np.random.randint(
                0, len(self.data_handlers))
            handler = self.data_handlers[handler_index]
            high_res = np.zeros((self.batch_size, self.sample_shape[0],
                                 self.sample_shape[1], self.shape[-1]),
                                dtype=np.float32)
            for i in range(self.batch_size):
                high_res[i, ...] = handler.get_next()[..., 0, :]

            batch = self.BATCH_CLASS.get_coarse_batch(
                high_res, self.s_enhance,
                output_features_ind=self.output_features_ind,
                training_features=self.training_features,
                smoothing=self.smoothing,
                smoothing_ignore=self.smoothing_ignore,
                model_mom1=self.model_mom1,
                s_padding=self.s_padding,
                t_padding=self.t_padding)

            self._i += 1
            return batch
        else:
            raise StopIteration


class ValidationDataMom1SF(ValidationDataMom1):
    """Iterator for validation data for
    first conditional moment of subfilter velocity"""
    BATCH_CLASS = BatchMom1SF


class ValidationDataMom2(ValidationDataMom1):
    """Iterator for subfilter validation data for
    second conditional moment"""
    BATCH_CLASS = BatchMom2


class ValidationDataMom2Sep(ValidationDataMom1):
    """Iterator for subfilter validation data for
    second conditional moment separate from first
    moment"""
    BATCH_CLASS = BatchMom2Sep


class ValidationDataMom2SF(ValidationDataMom1):
    """Iterator for validation data for
    second conditional moment of subfilter velocity"""
    BATCH_CLASS = BatchMom2SF


class ValidationDataMom2SepSF(ValidationDataMom1):
    """Iterator for validation data for
    second conditional moment of subfilter velocity
    separate from first moment"""
    BATCH_CLASS = BatchMom2SepSF


class BatchHandlerMom1SF(BatchHandlerMom1):
    """Sup3r batch handling class for
    first conditional moment of subfilter velocity"""
    VAL_CLASS = ValidationDataMom1SF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class SpatialBatchHandlerMom1SF(SpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for
    first conditional moment of subfilter velocity"""
    VAL_CLASS = ValidationDataMom1SF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class BatchHandlerMom2(BatchHandlerMom1):
    """Sup3r batch handling class for
    second conditional moment"""
    VAL_CLASS = ValidationDataMom2
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class BatchHandlerMom2Sep(BatchHandlerMom1):
    """Sup3r batch handling class for
    second conditional moment separate from first
    moment"""
    VAL_CLASS = ValidationDataMom2Sep
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class SpatialBatchHandlerMom2(SpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for
    second conditional moment"""
    VAL_CLASS = ValidationDataMom2
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class SpatialBatchHandlerMom2Sep(SpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for
    second conditional moment separate from first
    moment"""
    VAL_CLASS = ValidationDataMom2Sep
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class BatchHandlerMom2SF(BatchHandlerMom1):
    """Sup3r batch handling class for
    second conditional moment of subfilter velocity"""
    VAL_CLASS = ValidationDataMom2SF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class BatchHandlerMom2SepSF(BatchHandlerMom1):
    """Sup3r batch handling class for
    second conditional moment of subfilter velocity
    separate from first moment"""
    VAL_CLASS = ValidationDataMom2SepSF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class SpatialBatchHandlerMom2SF(SpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for
    second conditional moment of subfilter velocity"""
    VAL_CLASS = ValidationDataMom2SF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class SpatialBatchHandlerMom2SepSF(SpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for
    second conditional moment of subfilter velocity
    separate from first moment"""
    VAL_CLASS = ValidationDataMom2SepSF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS
