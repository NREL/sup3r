"""Ensure correct data shapes for :class:`Rasterizer` objects."""

import os
from tempfile import TemporaryDirectory

from sup3r.preprocessing import Rasterizer
from sup3r.utilities.pytest.helpers import make_fake_nc_file

features = ['windspeed_100m', 'winddirection_100m']
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
        rasterizer = Rasterizer([wind_file, level_file])
        assert rasterizer.shape == (10, 10, 20, 3, 5)
        assert sorted(rasterizer.features) == sorted(
            ['topography', 'u_100m', 'v_100m', 'zg', 'u']
        )
        assert rasterizer['u_100m'].shape == (10, 10, 20)
        assert rasterizer['U'].shape == (10, 10, 20, 3)
