"""Output method tests"""
import numpy as np

from sup3r.postprocessing.file_handling import OutputHandlerNC, OutputHandlerH5
from sup3r.utilities.utilities import invert_uv, transform_rotate_wind


def test_get_lat_lon():
    """Check that regridding works correctly"""
    low_res_lats = np.array([[1, 1, 1], [0, 0, 0]])
    low_res_lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate([np.expand_dims(low_res_lats, axis=-1),
                              np.expand_dims(low_res_lons, axis=-1)], axis=-1)
    shape = (4, 6)
    new_lat_lon = OutputHandlerNC.get_lat_lon(lat_lon, shape)

    new_lats = np.round(new_lat_lon[..., 0], 2)
    new_lons = np.round(new_lat_lon[..., 1], 2)

    assert np.allclose(new_lats[0, :], 1.25)
    assert np.allclose(new_lats[-1, :], -0.25)

    assert np.allclose(new_lons[:, 0], -125)
    assert np.allclose(new_lons[:, -1], -75)


def test_invert_uv():
    """Make sure inverse uv transform returns inputs"""
    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate([np.expand_dims(lats, axis=-1),
                              np.expand_dims(lons, axis=-1)], axis=-1)
    windspeed = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)
    winddirection = 360 * np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)

    u, v = transform_rotate_wind(np.array(windspeed, dtype=np.float32),
                                 np.array(winddirection, dtype=np.float32),
                                 lat_lon)

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(windspeed, ws)
    assert np.allclose(winddirection, wd)


def test_invert_uv_inplace():
    """Make sure inverse uv transform in output handler returns same as direct
    transform"""

    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate([np.expand_dims(lats, axis=-1),
                              np.expand_dims(lons, axis=-1)], axis=-1)
    u = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)
    v = np.random.rand(lat_lon.shape[0], lat_lon.shape[1], 5)

    data = np.concatenate([np.expand_dims(u, axis=-1),
                           np.expand_dims(v, axis=-1)], axis=-1)

    OutputHandlerH5.invert_uv_features(data, ['U_100m', 'V_100m'], lat_lon)

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(data[..., 0], ws)
    assert np.allclose(data[..., 1], wd)
