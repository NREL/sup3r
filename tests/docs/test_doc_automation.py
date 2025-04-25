"""Test for signature and doc automation for composite and dervied objects with
args / kwargs pass throughs"""

from inspect import signature

import pytest
from numpydoc.docscrape import NumpyDocString

from sup3r.preprocessing import (
    BatchHandlerDC,
    BatchQueueDC,
    DataHandler,
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
    DataHandlerNCforCC,
    ExoRasterizer,
    Rasterizer,
    SamplerDC,
)


@pytest.mark.parametrize(
    'obj',
    (
        BatchHandlerDC,
        DataHandler,
        BatchQueueDC,
        SamplerDC,
        DataHandlerNCforCC,
        DataHandlerH5SolarCC,
        DataHandlerH5WindCC,
        Rasterizer,
        ExoRasterizer
    ),
)
def test_full_docs(obj):
    """Make sure each arg in obj signature has an entry in the doc string."""

    sig = signature(obj)
    doc = obj.__init__.__doc__
    doc = doc if doc else obj.__doc__
    doc = NumpyDocString(doc)
    params = {p.name for p in sig.parameters.values()}
    doc_params = {p.name for p in doc['Parameters']}
    assert not params - doc_params


def test_h5_solar_sig():
    """Make sure signature of composite H5 solar data handler is resolved.

    This is a bad test, with hardcoded arg names, but I'm not sure of a better
    way here.
    """

    arg_names = [
        'file_paths',
        'features',
        'res_kwargs',
        'chunks',
        'target',
        'shape',
        'time_slice',
        'threshold',
        'time_roll',
        'hr_spatial_coarsen',
        'nan_method_kwargs',
        'interp_kwargs',
        'cache_kwargs'
    ]
    sig = signature(DataHandlerH5SolarCC)
    params = [p.name for p in sig.parameters.values()]
    assert not set(arg_names) - set(params)


def test_bh_sig():
    """Make sure signature of composite batch handler is resolved."""

    arg_names = [
        'train_containers',
        'sample_shape',
        'val_containers',
        'means',
        'stds',
        'feature_sets',
        'n_batches',
        't_enhance',
        'batch_size',
        'spatial_weights',
        'temporal_weights',
    ]
    sig = signature(BatchHandlerDC)
    init_sig = signature(BatchHandlerDC.__init__)
    params = [p.name for p in sig.parameters.values()]
    init_params = [p.name for p in init_sig.parameters.values()]
    assert not set(arg_names) - set(params)
    assert not set(arg_names) - set(init_params)


def test_nc_for_cc_sig():
    """Make sure signature of DataHandlerNCforCC is resolved."""
    arg_names = [
        'file_paths',
        'features',
        'nsrdb_source_fp',
        'nsrdb_agg',
        'nsrdb_smoothing',
        'shape',
        'target',
        'time_slice',
        'time_roll',
        'threshold',
        'hr_spatial_coarsen',
        'res_kwargs',
        'cache_kwargs',
        'chunks',
        'interp_kwargs',
        'nan_method_kwargs',
        'BaseLoader',
        'FeatureRegistry',
    ]
    sig = signature(DataHandlerNCforCC)
    init_sig = signature(DataHandlerNCforCC.__init__)
    params = [p.name for p in sig.parameters.values()]
    init_params = [p.name for p in init_sig.parameters.values()]
    assert not set(arg_names) - set(params)
    assert not set(arg_names) - set(init_params)


def test_dh_signature():
    """Make sure signature of composite data handler is resolved."""
    arg_names = [
        'file_paths',
        'features',
        'shape',
        'target',
        'time_slice',
        'time_roll',
        'threshold',
        'hr_spatial_coarsen',
        'res_kwargs',
        'cache_kwargs',
        'BaseLoader',
        'FeatureRegistry',
        'chunks',
        'interp_kwargs',
        'nan_method_kwargs'
    ]
    sig = signature(DataHandler)
    init_sig = signature(DataHandler.__init__)
    params = [p.name for p in sig.parameters.values()]
    init_params = [p.name for p in init_sig.parameters.values()]
    assert not set(arg_names) - set(params)
    assert not set(arg_names) - set(init_params)
