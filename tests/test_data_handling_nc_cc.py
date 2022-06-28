"""Test data handler for netcdf climate change data"""
import os
import numpy as np
import xarray as xr

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import DataHandlerNCforCC


def test_data_handling_nc_cc():
    """Make sure the netcdf cc data handler operates correctly"""

    input_files = [os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'va_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'zg_test.nc')]

    with xr.open_mfdataset(input_files) as fh:
        min_lat = np.min(fh.lat.values)
        min_lon = np.min(fh.lon.values)
        target = (min_lat, min_lon)
        plevel = fh.plev[-1]
        ua = np.transpose(fh['ua'][:, -1, ...].values, (1, 2, 0))
        va = np.transpose(fh['va'][:, -1, ...].values, (1, 2, 0))

    handler = DataHandlerNCforCC(input_files, features=['U_100m', 'V_100m'],
                                 target=target, shape=(20, 20),
                                 val_split=0.0)

    assert handler.data.shape == (20, 20, 20, 2)

    handler = DataHandlerNCforCC(input_files,
                                 features=[f'U_{int(plevel)}pa',
                                           f'V_{int(plevel)}pa'],
                                 target=target, shape=(20, 20),
                                 val_split=0.0)

    assert handler.data.shape == (20, 20, 20, 2)

    assert np.allclose(ua, handler.data[..., 0])
    assert np.allclose(va, handler.data[..., 1])
