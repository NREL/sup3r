"""Test data handler for netcdf climate change data"""
import os

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing.data_handling import DataHandlerNCforCC


def test_data_handling_nc_cc():
    """Make sure the netcdf cc data handler operates correctly"""

    input_files = [os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'va_test.nc'),
                   os.path.join(TEST_DATA_DIR, 'zg_test.nc')]
    handler = DataHandlerNCforCC(input_files, features=['U_100m', 'V_100m'],
                                 target=(-90, 0), shape=(20, 20),
                                 val_split=0.0)

    assert handler.data.shape == (20, 20, 20, 2)

    handler = DataHandlerNCforCC(input_files,
                                 features=['U_10000pa', 'V_10000pa'],
                                 target=(-90, 0), shape=(20, 20),
                                 val_split=0.0)

    assert handler.data.shape == (20, 20, 20, 2)
