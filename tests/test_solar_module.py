# -*- coding: utf-8 -*-
"""Test the custom sup3r solar module that converts GAN clearsky ratio outputs
to irradiance data."""
import os
import numpy as np
import tempfile
import matplotlib.pyplot as plt

from sup3r import TEST_DATA_DIR
from sup3r.solar import Solar
from sup3r.postprocessing.file_handling import OutputHandlerH5
from sup3r.utilities.utilities import pd_date_range


def test_solar_module(plot=False):
    """Test the solar module operating on a set of SolarMultiStepGan chunked
    outputs"""

    nsrdb_fp = os.path.join(TEST_DATA_DIR, 'test_nsrdb_clearsky_2018.h5')

    lr_lat = np.linspace(40, 39, 5)
    lr_lon = np.linspace(-105.5, -104.3, 5)
    lr_lon, lr_lat = np.meshgrid(lr_lon, lr_lat)
    lr_lon = np.expand_dims(lr_lon, axis=2)
    lr_lat = np.expand_dims(lr_lat, axis=2)
    low_res_lat_lon = np.concatenate((lr_lat, lr_lon), axis=2)

    low_res_times = pd_date_range('20180101', '20180104',
                                  inclusive='left', freq='1d')

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
                                         out_file, max_workers=1)
            fps.append(out_file)

        with Solar(fps, nsrdb_fp) as solar:
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

            # make sure the roll worked and the overflow is back filled instead
            # of leaving the roll seam.
            for i in range(6):
                assert np.allclose(solar.clearsky_ratio[i, :],
                                   solar.clearsky_ratio[6, :])

        if plot:
            for i in range(len(ghi)):
                ghi_raster = ghi[i].reshape((20, 20))
                a = plt.imshow(ghi_raster, vmin=0, vmax=cs_ghi.max())
                plt.colorbar(a)
                plt.savefig('./test_ghi_{}.png'.format(i))
                plt.close()
