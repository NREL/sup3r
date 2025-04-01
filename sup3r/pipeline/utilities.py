"""Methods used by :class:`ForwardPass` and :class:`ForwardPassStrategy`"""
import logging

import numpy as np

import sup3r.models

logger = logging.getLogger(__name__)


def get_model(model_class, kwargs):
    """Instantiate model after check on class name."""
    msg = (
        'Could not load requested model class "{}" from '
        'sup3r.models, Make sure you typed in the model class '
        'name correctly.'.format(model_class)
    )
    model_class = getattr(sup3r.models, model_class, None)
    if isinstance(kwargs, str):
        kwargs = {'model_dir': kwargs}
    if model_class is None:
        logger.error(msg)
        raise KeyError(msg)
    return model_class.load(**kwargs, verbose=True)


def get_chunk_slices(arr_size, chunk_size, index_slice=slice(None)):
    """Get array slices of corresponding chunk size

    Parameters
    ----------
    arr_size : int
        Length of array to slice
    chunk_size : int
        Size of slices to split array into
    index_slice : slice
        Slice specifying starting and ending index of slice list

    Returns
    -------
    list
        List of slices corresponding to chunks of array
    """

    indices = np.arange(0, arr_size)
    indices = indices[slice(index_slice.start, index_slice.stop)]
    step = 1 if index_slice.step is None else index_slice.step
    slices = []
    start = indices[0]
    stop = start + step * chunk_size
    stop = np.min([stop, indices[-1] + 1])

    while start < indices[-1] + 1:
        slices.append(slice(start, stop, step))
        start = stop
        stop += step * chunk_size
        stop = np.min([stop, indices[-1] + 1])
    return slices
