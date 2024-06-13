"""Ensure correct data shapes for :class:`Extracter` objects."""

import os
from tempfile import TemporaryDirectory

from rex import init_logger

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import ExtracterNC
from sup3r.utilities.pytest.helpers import execute_pytest, make_fake_nc_file

h5_files = [
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5'),
    os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
]
nc_files = [os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')]

features = ['windspeed_100m', 'winddirection_100m']

init_logger('sup3r', log_level='DEBUG')

h5_target = (39.01, -105.15)
nc_target = (37.25, -107)
h5_shape = (20, 20)
nc_shape = (10, 10)


def test_5d_extract_nc():
    """Test loading netcdf data with some multi level features."""
    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(
            wind_file,
            shape=(10, 10, 20),
            features=['orog', 'u_100m', 'v_100m'],
        )
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )
        extracter = ExtracterNC([wind_file, level_file])
        assert extracter.shape == (10, 10, 20, 3, 5)
        assert sorted(extracter.features) == sorted(
            ['topography', 'u_100m', 'v_100m', 'zg', 'u']
        )
        assert extracter['U_100m'].shape == (10, 10, 20)
        assert extracter['U'].shape == (10, 10, 20, 3)


if __name__ == '__main__':
    execute_pytest(__file__)
