"""Test the custom sup3r solar module that converts GAN clearsky ratio outputs
to irradiance data."""
import glob
import json
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner
from rex import Resource

from sup3r import TEST_DATA_DIR
from sup3r.solar import Solar
from sup3r.solar.solar_cli import from_config as solar_main
from sup3r.utilities.pytest.helpers import make_fake_cs_ratio_files
from sup3r.utilities.utilities import pd_date_range

NSRDB_FP = os.path.join(TEST_DATA_DIR, 'test_nsrdb_clearsky_2018.h5')
GAN_META = {'s_enhance': 4, 't_enhance': 24}
LR_LAT = np.linspace(40, 39, 5)
LR_LON = np.linspace(-105.5, -104.3, 5)
LR_LON, LR_LAT = np.meshgrid(LR_LON, LR_LAT)
LR_LON = np.expand_dims(LR_LON, axis=2)
LR_LAT = np.expand_dims(LR_LAT, axis=2)
LOW_RES_LAT_LON = np.concatenate((LR_LAT, LR_LON), axis=2)
LOW_RES_TIMES = pd_date_range('20500101', '20500104',
                              inclusive='left', freq='1d')
HIGH_RES_TIMES = pd_date_range('20500101', '20500104',
                               inclusive='left', freq='1h')


@pytest.fixture(scope='module')
def runner():
    """Cli runner helper utility."""
    return CliRunner()


def test_solar_module(plot=False):
    """Test the solar module operating on a set of SolarMultiStepGan chunked
    outputs"""

    t_slice = slice(24, 48)

    with tempfile.TemporaryDirectory() as td:

        fps, _ = make_fake_cs_ratio_files(td, LOW_RES_TIMES, LOW_RES_LAT_LON,
                                          model_meta=GAN_META)

        with Resource(fps[1]) as res:
            meta_base = res.meta
            attrs_base = res.global_attrs

        with Solar(fps, NSRDB_FP, t_slice=t_slice,
                   nn_threshold=0.4) as solar:
            ghi = solar.ghi
            dni = solar.dni
            dhi = solar.dhi
            cs_ghi = solar.get_nsrdb_data('clearsky_ghi')
            cs_dni = solar.get_nsrdb_data('clearsky_dni')

            # check solar irrad limits
            assert (ghi <= cs_ghi).all()
            assert (dni <= cs_dni).all()
            assert (ghi >= 0).all()
            assert (dni >= 0).all()
            assert (dhi >= 0).all()
            assert 0.3 < ((ghi > 0).sum() / ghi.size) < 0.4
            assert 0.2 < ((dni > 0).sum() / dni.size) < 0.3

            # check that some pixels are out of bounds
            assert 10 < solar.out_of_bounds.sum() < 30
            assert (ghi[:, solar.out_of_bounds] == 0).all()
            assert (dni[:, solar.out_of_bounds] == 0).all()
            assert (dhi[:, solar.out_of_bounds] == 0).all()

            fp_out = os.path.join(td, 'solar/out.h5')
            solar.write(fp_out)

            assert os.path.exists(fp_out)
            with Resource(fp_out) as res:
                assert np.allclose(res['ghi'], ghi, atol=1)
                assert np.allclose(res['dni'], dni, atol=1)
                assert np.allclose(res['dhi'], dhi, atol=1)

                res_ti = res.time_index.tz_convert(None)
                assert (HIGH_RES_TIMES[t_slice] == res_ti).all()
                assert np.allclose(res.meta['latitude'],
                                   meta_base['latitude'])
                assert np.allclose(res.meta['longitude'],
                                   meta_base['longitude'])

                gattrs = res.global_attrs
                assert gattrs['nsrdb_source'] == NSRDB_FP
                assert gattrs['t_enhance'] == str(GAN_META['t_enhance'])
                assert gattrs['s_enhance'] == str(GAN_META['s_enhance'])
                for k, v in attrs_base.items():
                    assert gattrs[k] == v

        if plot:
            for i, timestamp in enumerate(HIGH_RES_TIMES[t_slice]):
                ghi_raster = ghi[i].reshape((20, 20))
                a = plt.imshow(ghi_raster, vmin=0, vmax=cs_ghi.max())
                plt.colorbar(a)
                plt.savefig('./test_ghi_{}.png'.format(timestamp))
                plt.close()


def test_chunk_file_parser():
    """Test the solar utility that retrieves the fwp chunked output file sets
    to be run."""
    id_temporal = [str(i).zfill(6) for i in range(4, 7)]
    id_spatial = [str(i).zfill(6) for i in range(6, 10)]
    all_st_ids = []
    all_fps = []
    with tempfile.TemporaryDirectory() as td:
        for idt in id_temporal:
            for ids in id_spatial:
                fn = 'sup3r_chunk_out_{}_{}.h5'.format(idt, ids)
                fp = os.path.join(td, fn)
                Path(fp).touch()
                all_st_ids.append('{}_{}'.format(idt, ids))
                all_fps.append(fp)

        fp_pattern = os.path.join(td, 'sup3r_chunk_*.h5')
        fp_sets, t_slices, _, _, _ = Solar.get_sup3r_fps(fp_pattern)

        for fp_set, t_slice in zip(fp_sets, t_slices):
            s_ids = [os.path.basename(fp).replace('.h5', '').split('_')[-1]
                     for fp in fp_set]
            t_ids = [os.path.basename(fp).replace('.h5', '').split('_')[-2]
                     for fp in fp_set]

            assert len(set(s_ids)) == 1

            if t_slice.start == 0:
                assert id_temporal.index(t_ids[0]) == 0
                assert id_temporal.index(t_ids[-1]) == 1

            if id_temporal.index(t_ids[0]) == len(id_temporal) - 2:
                assert len(fp_set) == 2


def test_solar_cli(runner):
    """Test the solar CLI. This test is here and not in the test_cli.py file
    because it uses some common test utilities stored here."""
    with tempfile.TemporaryDirectory() as td:
        fps, fp_pattern = make_fake_cs_ratio_files(td, LOW_RES_TIMES,
                                                   LOW_RES_LAT_LON,
                                                   model_meta=GAN_META)
        config = {'fp_pattern': fp_pattern,
                  'nsrdb_fp': NSRDB_FP,
                  'log_level': 'DEBUG',
                  'log_pattern': os.path.join(td, 'logs/solar.log')}

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(solar_main, ['-c', config_path, '-v'])

        if result.exit_code != 0:
            import traceback
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))

            log_file = os.path.join(td, 'logs/sup3r_solar.log')
            if os.path.exists(log_file):
                with open(log_file) as f:
                    logs = ''.join(list(f.readlines()))
                msg += '\nlogs:\n{}'.format(logs)

            raise RuntimeError(msg)

        status_files = glob.glob(os.path.join(f'{td}/.gaps/',
                                              '*jobstatus*.json'))
        assert len(status_files) == len(fps)

        out_files = glob.glob(os.path.join(td, 'chunks/*_irradiance.h5'))
        assert len(out_files) == len(fps)
