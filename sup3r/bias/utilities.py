"""Bias correction methods which can be applied to data handler data."""
import logging
import os

import numpy as np
from rex import Resource

from sup3r.bias.bias_transforms import get_spatial_bc_factors, local_qdm_bc

logger = logging.getLogger(__name__)


def lin_bc(handler, bc_files, threshold=0.1):
    """Bias correct the data in this DataHandler using linear bias
    correction factors from files output by MonthlyLinearCorrection or
    LinearCorrection from sup3r.bias.bias_calc

    Parameters
    ----------
    handler : DataHandler
        DataHandler instance with `.data` attribute containing data to
        bias correct
    bc_files : list | tuple | str
        One or more filepaths to .h5 files output by
        MonthlyLinearCorrection or LinearCorrection. These should contain
        datasets named "{feature}_scalar" and "{feature}_adder" where
        {feature} is one of the features contained by this DataHandler and
        the data is a 3D array of shape (lat, lon, time) where time is
        length 1 for annual correction or 12 for monthly correction.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the DataHandler
        coordinates are more than this value away from the bias correction
        lat/lon, an error is raised.
    """

    if isinstance(bc_files, str):
        bc_files = [bc_files]

    completed = []
    for idf, feature in enumerate(handler.features):
        for fp in bc_files:
            dset_scalar = f'{feature}_scalar'
            dset_adder = f'{feature}_adder'
            with Resource(fp) as res:
                dsets = [dset.lower() for dset in res.dsets]
                check = (dset_scalar.lower() in dsets
                         and dset_adder.lower() in dsets)
            if feature not in completed and check:
                scalar, adder = get_spatial_bc_factors(
                    lat_lon=handler.lat_lon,
                    feature_name=feature,
                    bias_fp=fp,
                    threshold=threshold)

                if scalar.shape[-1] == 1:
                    scalar = np.repeat(scalar, handler.shape[2], axis=2)
                    adder = np.repeat(adder, handler.shape[2], axis=2)
                elif scalar.shape[-1] == 12:
                    idm = handler.time_index.month.values - 1
                    scalar = scalar[..., idm]
                    adder = adder[..., idm]
                else:
                    msg = ('Can only accept bias correction factors '
                           'with last dim equal to 1 or 12 but '
                           'received bias correction factors with '
                           'shape {}'.format(scalar.shape))
                    logger.error(msg)
                    raise RuntimeError(msg)

                logger.info('Bias correcting "{}" with linear '
                            'correction from "{}"'.format(
                                feature, os.path.basename(fp)))
                handler.data[..., idf] *= scalar
                handler.data[..., idf] += adder
                completed.append(feature)


def qdm_bc(handler,
           bc_files,
           reference_feature,
           relative=True,
           threshold=0.1,
           no_trend=False):
    """Bias Correction using Quantile Delta Mapping

    Bias correct this DataHandler's data with Quantile Delta Mapping. The
    required statistical distributions should be pre-calculated using
    :class:`sup3r.bias.bias_calc.QuantileDeltaMappingCorrection`.

    Warning: There is no guarantee that the coefficients from ``bc_files``
    match the resource processed here. Be careful choosing ``bc_files``.

    Parameters
    ----------
    handler : DataHandler
        DataHandler instance with `.data` attribute containing data to
        bias correct
    bc_files : list | tuple | str
        One or more filepaths to .h5 files output by
        :class:`bias_calc.QuantileDeltaMappingCorrection`. These should
        contain datasets named "base_{reference_feature}_params",
        "bias_{feature}_params", and "bias_fut_{feature}_params" where
        {feature} is one of the features contained by this DataHandler and
        the data is a 3D array of shape (lat, lon, time) where time.
    reference_feature : str
        Name of the feature used as (historical) reference. Dataset with
        name "base_{reference_feature}_params" will be retrieved from
        ``bc_files``.
    relative : bool, default=True
        Switcher to apply QDM as a relative (use True) or absolute (use
        False) correction value.
    threshold : float, default=0.1
        Nearest neighbor euclidean distance threshold. If the DataHandler
        coordinates are more than this value away from the bias correction
        lat/lon, an error is raised.
    no_trend: bool, default=False
        An option to ignore the trend component of the correction, thus
        resulting in an ordinary Quantile Mapping, i.e. corrects the bias
        by comparing the distributions of the biased dataset with a
        reference datasets. See ``params_mf`` of
        :class:`rex.utilities.bc_utils.QuantileDeltaMapping`.
        Note that this assumes that "bias_{feature}_params"
        (``params_mh``) is the data distribution representative for the
        target data.
    """

    if isinstance(bc_files, str):
        bc_files = [bc_files]

    completed = []
    for idf, feature in enumerate(handler.features):
        for fp in bc_files:
            logger.info('Bias correcting "{}" with QDM '
                        'correction from "{}"'.format(
                            feature, os.path.basename(fp)))
            handler.data[..., idf] = local_qdm_bc(handler.data[..., idf],
                                               handler.lat_lon,
                                               reference_feature,
                                               feature,
                                               bias_fp=fp,
                                               threshold=threshold,
                                               relative=relative,
                                               no_trend=no_trend)
            completed.append(feature)
