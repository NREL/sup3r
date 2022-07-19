"""Utilities used across multiple test files"""

import os
import numpy as np
import xarray as xr


def make_fake_nc_files(td, input_file, n_files):
    """Make dummy nc files with increasing times

    Parameters
    ----------
    input_file : str
        File to use as template for all dummy files
    n_files : int
        Number of dummy files to create

    Returns
    -------
    fake_files : list
        List of dummy files
    """
    fake_dates = [f'2014-10-01_0{i}_00_00' for i in range(n_files)]
    fake_times = [f'2014-10-01 0{i}:00:00' for i in range(n_files)]
    fake_files = [os.path.join(td, f'input_{date}') for date in fake_dates]
    for i in range(n_files):
        input_dset = xr.open_dataset(input_file)
        with xr.Dataset(input_dset) as dset:
            dset['Times'][:] = np.array([fake_times[i].encode('ASCII')],
                                        dtype='|S19')
            dset['XTIME'][:] = i
            dset.to_netcdf(fake_files[i])
    return fake_files
