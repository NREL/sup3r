"""Test the training of super resolution GANs with exogenous observation
data."""

import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGanFixedObs
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.utilities.pytest.helpers import make_fake_dset, make_fake_nc_file
from sup3r.utilities.utilities import RANDOM_GENERATOR

SHAPE = (20, 20)
FEATURES_W = ['u_10m', 'v_10m']
TARGET_W = (39.01, -105.15)

target = (19.3, -123.5)
shape = (8, 8)
sample_shape = (8, 8, 6)
time_slice = slice(None, None, 1)
list_chunk_size = 10
fwp_chunk_shape = (4, 4, 150)
s_enhance = 3
t_enhance = 4


@pytest.fixture(scope='module')
def input_files(tmpdir_factory):
    """Dummy netcdf input files for :class:`ForwardPass`"""

    input_file = str(tmpdir_factory.mktemp('data').join('fwp_input.nc'))
    make_fake_nc_file(
        input_file,
        shape=(100, 100, 8),
        features=['u_10m', 'v_10m'],
    )
    obs_file = str(tmpdir_factory.mktemp('data').join('fwp_obs.nc'))
    dset = make_fake_dset(
        shape=(100, 100, 8),
        features=['u_10m', 'v_10m'],
    )

    mask = RANDOM_GENERATOR.choice(
        [True, False], dset['u_10m'].shape, p=[0.9, 0.1]
    )
    dset['u_10m'][mask] = np.nan
    dset['v_10m'][mask] = np.nan
    dset.to_netcdf(obs_file)

    return input_file, obs_file


def test_fwp_with_obs(input_files, gen_config_with_concat_masked):
    """Test a special model trained to conditional output on input
    observations."""

    Sup3rGanFixedObs.seed()

    model = Sup3rGanFixedObs(
        gen_config_with_concat_masked(),
        pytest.S_FP_DISC,
        obs_frac={'spatial': 0.1},
        loss_obs_weight=0.1,
        learning_rate=1e-4,
        input_resolution={'spatial': '16km', 'temporal': '3600min'},
    )

    model.meta['lr_features'] = ['u_10m', 'v_10m']
    model.meta['hr_out_features'] = ['u_10m', 'v_10m']
    model.meta['s_enhance'] = 2
    model.meta['t_enhance'] = 1

    with tempfile.TemporaryDirectory() as td:
        model_dir = os.path.join(td, 'test')
        model.save(model_dir)

        exo_handler_kwargs = {
            'u_10m': {
                'file_paths': input_files[0],
                'source_file': input_files[1],
                'target': target,
                'shape': shape,
                'cache_dir': td,
            },
            'v_10m': {
                'file_paths': input_files[0],
                'source_file': input_files[1],
                'target': target,
                'shape': shape,
                'cache_dir': td,
            },
        }

        model_kwargs = {'model_dirs': [model_dir]}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            input_handler_kwargs=input_handler_kwargs,
            spatial_pad=0,
            temporal_pad=0,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            max_nodes=1,
            pass_workers=2,
        )

        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)
