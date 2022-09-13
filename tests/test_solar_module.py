# -*- coding: utf-8 -*-
"""Test the custom sup3r solar module that converts GAN clearsky ratio outputs
to irradiance data."""
import os
import numpy as np
import tempfile

from sup3r.postprocessing.file_handling import OutputHandlerH5
from sup3r.utilities.utilities import pd_date_range

FEATURES = ['U_100m', 'V_100m']


if __name__ == '__main__':
    """Test the solar module operating on a set of SolarMultiStepGan chunked
    outputs"""

    lr_lat = np.linspace(25, 20, 5)
    lr_lon = np.linspace(-110, -100, 5)
    lr_lon, lr_lat = np.meshgrid(lr_lon, lr_lat)
    lr_lon = np.expand_dims(lr_lon, axis=2)
    lr_lat = np.expand_dims(lr_lat, axis=2)
    low_res_lat_lon = np.concatenate((lr_lat, lr_lon), axis=2)

    low_res_times = pd_date_range('20200101', '20200104',
                                  inclusive='left', freq='1d')

    fps = []

    with tempfile.TemporaryDirectory() as td:
        chunk_dir = os.path.join(td, 'chunks/')
        os.makedirs(chunk_dir)

        for idt, timestamp in enumerate(low_res_times):
            fn = ('sup3r_chunk_{}_{}.h5'
                  .format(str(idt).zfill(6), str(0).zfill(6)))
            out_file = os.path.join(chunk_dir, fn)

            cs_ratio = np.random.uniform(0, 1, (20, 20, 24, 1))

            OutputHandlerH5.write_output(cs_ratio, ['clearsky_ratio'],
                                         low_res_lat_lon,
                                         [timestamp],
                                         out_file, max_workers=1)
            fps.append(out_file)
