"""Utilities used across multiple test files"""

import shutil
import os
from netCDF4 import Dataset


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
    fake_dates = [f'2014-10-01_0{i}_00_00' for i in range(8)]
    fake_times = list(range(n_files))

    fake_files = [os.path.join(td, f'input_{date}') for date in fake_dates]
    for i in range(n_files):
        shutil.copy(input_file, fake_files[i])
        with Dataset(fake_files[i], 'r+') as dset:
            dset['XTIME'][:] = fake_times[i]
    return fake_files
