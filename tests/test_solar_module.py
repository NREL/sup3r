# -*- coding: utf-8 -*-
"""Test the custom sup3r solar module that converts GAN clearsky ratio outputs
to irradiance data."""
import json
import os
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from rex import Resource

from sup3r import TEST_DATA_DIR
from sup3r.solar import Solar
from sup3r.postprocessing.file_handling import OutputHandlerH5
from sup3r.utilities.utilities import pd_date_range


def test_solar_module(plot=False):
    """Test the solar module operating on a set of SolarMultiStepGan chunked
    outputs"""

    nsrdb_fp = os.path.join(TEST_DATA_DIR, 'test_nsrdb_clearsky_2018.h5')

    gan_meta = {'s_enhance': 4, 't_enhance': 24}

    lr_lat = np.linspace(40, 39, 5)
    lr_lon = np.linspace(-105.5, -104.3, 5)
    lr_lon, lr_lat = np.meshgrid(lr_lon, lr_lat)
    lr_lon = np.expand_dims(lr_lon, axis=2)
    lr_lat = np.expand_dims(lr_lat, axis=2)
    low_res_lat_lon = np.concatenate((lr_lat, lr_lon), axis=2)

    t_slice = slice(24, 48)
    low_res_times = pd_date_range('20180101', '20180104',
                                  inclusive='left', freq='1d')
    high_res_times = pd_date_range('20180101', '20180104',
                                   inclusive='left', freq='1h')

    fps = []

    with tempfile.TemporaryDirectory() as td:
        chunk_dir = os.path.join(td, 'chunks/')
        os.makedirs(chunk_dir)

        for idt, timestamp in enumerate(low_res_times):
            fn = ('sup3r_chunk_{}_{}.h5'
                  .format(str(idt).zfill(6), str(0).zfill(6)))
            out_file = os.path.join(chunk_dir, fn)

            cs_ratio = np.random.uniform(0, 1, (20, 20, 1, 1))
            cs_ratio = np.repeat(cs_ratio, 24, axis=2)

            OutputHandlerH5.write_output(cs_ratio, ['clearsky_ratio'],
                                         low_res_lat_lon,
                                         [timestamp],
                                         out_file, max_workers=1,
                                         meta_data=gan_meta)
            fps.append(out_file)

        with Resource(fps[1]) as res:
            meta_base = res.meta
            attrs_base = res.global_attrs

        with Solar(fps, nsrdb_fp, t_slice=t_slice, nn_threshold=0.4) as solar:
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
                assert (high_res_times[t_slice] == res_ti).all()
                assert np.allclose(res.meta['latitude'],
                                   meta_base['latitude'])
                assert np.allclose(res.meta['longitude'],
                                   meta_base['longitude'])

                assert res.global_attrs['nsrdb_source'] == nsrdb_fp
                assert res.global_attrs['gan_meta'] == json.dumps(gan_meta)
                for k, v in attrs_base.items():
                    assert res.global_attrs[k] == v

        if plot:
            for i, timestamp in enumerate(high_res_times[t_slice]):
                ghi_raster = ghi[i].reshape((20, 20))
                a = plt.imshow(ghi_raster, vmin=0, vmax=cs_ghi.max())
                plt.colorbar(a)
                plt.savefig('./test_ghi_{}.png'.format(timestamp))
                plt.close()
