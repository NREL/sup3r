# -*- coding: utf-8 -*-
"""
Sup3r wind conditional moment batch_handling module.
"""
import logging

import numpy as np
import tensorflow as tf

from sup3r.preprocessing.batch_handling import Batch
from sup3r.preprocessing.conditional_moment_batch_handling import (
    BatchHandlerMom1,
    BatchMom1,
    SpatialBatchHandlerMom1,
    ValidationDataMom1,
)
from sup3r.utilities.utilities import (
    spatial_simple_enhancing,
    temporal_simple_enhancing,
)

np.random.seed(42)

logger = logging.getLogger(__name__)


class WindBatchMom1(BatchMom1):
    """Batch of low_res, high_res and output wind data"""


class WindBatchMom1SF(WindBatchMom1):
    """Batch of low_res, high_res and output wind data when learning first
    moment of subfilter vel"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None,
                    t_enhance_mode='constant'):
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
        t_enhance_mode : str
            Enhancing mode for temporal subfilter.
            Can be either constant or linear

        Returns
        -------
        SF: np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR from HR
        enhanced_lr = spatial_simple_enhancing(low_res,
                                               s_enhance=s_enhance)
        enhanced_lr = temporal_simple_enhancing(enhanced_lr,
                                                t_enhance=t_enhance,
                                                mode=t_enhance_mode)
        enhanced_lr = Batch.reduce_features(enhanced_lr, output_features_ind)
        enhanced_lr[..., -1] = high_res[..., -1]

        return high_res - enhanced_lr


class WindBatchMom2(WindBatchMom1):
    """Batch of low_res, high_res and output wind data when learning second
    moment"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None,
                    t_enhance_mode='constant'):
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
        t_enhance_mode : str
            Enhancing mode for temporal subfilter.
            Can be either constant or linear

        Returns
        -------
        (HR - <HR|LR>)**2 : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            HR is high-res and LR is low-res
        """
        # Remove first moment from HR and square it
        out = model_mom1._tf_generate(
            low_res, {'topography': high_res[..., -1:]}).numpy()
        out = tf.concat((out, high_res[..., -1:]), axis=-1)
        return (high_res - out)**2


class WindBatchMom2Sep(WindBatchMom1):
    """Batch of low_res, high_res and output wind data when learning second
    moment separate from first moment"""


class WindBatchMom2SF(WindBatchMom1):
    """Batch of low_res, high_res and output wind data when learning second
    moment of subfilter vel"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None,
                    t_enhance_mode='constant'):
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
        t_enhance_mode : str
            Enhancing mode for temporal subfilter.
            Can be either 'constant' or 'linear'

        Returns
        -------
        (SF - <SF|LR>)**2 : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR and first moment from HR and square it
        out = model_mom1._tf_generate(
            low_res, {'topography': high_res[..., -1:]}).numpy()
        out = tf.concat((out, high_res[..., -1:]), axis=-1)
        enhanced_lr = spatial_simple_enhancing(low_res,
                                               s_enhance=s_enhance)
        enhanced_lr = temporal_simple_enhancing(enhanced_lr,
                                                t_enhance=t_enhance,
                                                mode=t_enhance_mode)
        enhanced_lr = Batch.reduce_features(enhanced_lr, output_features_ind)
        enhanced_lr[..., -1] = 0.0
        return (high_res - enhanced_lr - out)**2


class WindBatchMom2SepSF(WindBatchMom1SF):
    """Batch of low_res, high_res and output wind data when learning second
    moment of subfilter vel separate from first moment"""

    @staticmethod
    def make_output(low_res, high_res,
                    s_enhance=None, t_enhance=None,
                    model_mom1=None, output_features_ind=None,
                    t_enhance_mode='constant'):
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
        t_enhance_mode : str
            Enhancing mode for temporal subfilter.
            Can be either constant or linear

        Returns
        -------
        SF**2 : np.ndarray
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR from HR and square it
        return super(WindBatchMom2SepSF,
                     WindBatchMom2SepSF).make_output(low_res, high_res,
                                                     s_enhance, t_enhance,
                                                     model_mom1,
                                                     output_features_ind,
                                                     t_enhance_mode)**2


class WindBatchHandlerMom1(BatchHandlerMom1):
    """Sup3r base batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationDataMom1
    BATCH_CLASS = WindBatchMom1
    DATA_HANDLER_CLASS = None


class WindSpatialBatchHandlerMom1(SpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class"""

    # Classes to use for handling an individual batch obj.
    VAL_CLASS = ValidationDataMom1
    BATCH_CLASS = WindBatchMom1
    DATA_HANDLER_CLASS = None


class ValidationDataWindMom1SF(ValidationDataMom1):
    """Iterator for validation wind data for first conditional moment of
    subfilter velocity"""

    BATCH_CLASS = WindBatchMom1SF


class ValidationDataWindMom2(ValidationDataMom1):
    """Iterator for subfilter validation wind data for second conditional
    moment"""

    BATCH_CLASS = WindBatchMom2


class ValidationDataWindMom2Sep(ValidationDataMom1):
    """Iterator for subfilter validation wind data for second conditional
    moment separate from first moment"""

    BATCH_CLASS = WindBatchMom2Sep


class ValidationDataWindMom2SF(ValidationDataMom1):
    """Iterator for validation wind data for second conditional moment of
    subfilter velocity"""

    BATCH_CLASS = WindBatchMom2SF


class ValidationDataWindMom2SepSF(ValidationDataMom1):
    """Iterator for validation wind data for second conditional moment of
    subfilter velocity separate from first moment"""

    BATCH_CLASS = WindBatchMom2SepSF


class WindBatchHandlerMom1SF(WindBatchHandlerMom1):
    """Sup3r batch handling class for first conditional moment of subfilter
    velocity using topography as input"""

    VAL_CLASS = ValidationDataWindMom1SF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindSpatialBatchHandlerMom1SF(WindSpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for first conditional moment of
    subfilter velocity using topography as input"""

    VAL_CLASS = ValidationDataWindMom1SF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindBatchHandlerMom2(WindBatchHandlerMom1):
    """Sup3r batch handling class for second conditional moment using
    topography as input"""

    VAL_CLASS = ValidationDataWindMom2
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindBatchHandlerMom2Sep(WindBatchHandlerMom1):
    """Sup3r batch handling class for second conditional moment separate from
    first moment using topography as input"""

    VAL_CLASS = ValidationDataWindMom2Sep
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindSpatialBatchHandlerMom2(WindSpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for second conditional moment using
    topography as input"""

    VAL_CLASS = ValidationDataWindMom2
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindSpatialBatchHandlerMom2Sep(WindSpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for second conditional moment
    separate from first moment using topography as input"""

    VAL_CLASS = ValidationDataWindMom2Sep
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindBatchHandlerMom2SF(WindBatchHandlerMom1):
    """Sup3r batch handling class for second conditional moment of subfilter
    velocity"""

    VAL_CLASS = ValidationDataWindMom2SF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindBatchHandlerMom2SepSF(WindBatchHandlerMom1):
    """Sup3r batch handling class for second conditional moment of subfilter
    velocity separate from first moment using topography as input"""

    VAL_CLASS = ValidationDataWindMom2SepSF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindSpatialBatchHandlerMom2SF(WindSpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for second conditional moment of
    subfilter velocity using topography as input"""

    VAL_CLASS = ValidationDataWindMom2SF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS


class WindSpatialBatchHandlerMom2SepSF(WindSpatialBatchHandlerMom1):
    """Sup3r spatial batch handling class for second conditional moment of
    subfilter velocity separate from first moment using topography as input"""

    VAL_CLASS = ValidationDataWindMom2SepSF
    BATCH_CLASS = VAL_CLASS.BATCH_CLASS
