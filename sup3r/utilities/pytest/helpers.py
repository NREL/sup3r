"""Testing helpers."""

import os

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sup3r.containers.abstract import AbstractContainer, DataWrapper
from sup3r.containers.samplers import CroppedSampler, Sampler
from sup3r.postprocessing.file_handling import OutputHandlerH5
from sup3r.utilities.utilities import pd_date_range


def execute_pytest(fname, capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    fname : str
        test file to run
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


def make_fake_dset(shape, features):
    """Make dummy data for tests."""
    times = pd.date_range('2023-01-01', '2023-12-31', freq='60min')[: shape[0]]

    if len(shape) == 3:
        dims = ('time', 'latitude', 'longitude')
        lats = np.linspace(70, -70, shape[1])
        lons = np.linspace(-150, 150, shape[2])
        coords = {'time': times, 'latitude': lats, 'longitude': lons}

    if len(shape) == 4:
        dims = ('time', 'level', 'latitude', 'longitude')
        levels = np.linspace(0, 1000, shape[1])
        lats = np.linspace(70, -70, shape[2])
        lons = np.linspace(-150, 150, shape[3])
        coords = {
            'time': times,
            'level': levels,
            'latitude': lats,
            'longitude': lons,
        }

    data_vars = {f: (dims, da.random.random(shape)) for f in features}
    nc = xr.Dataset(coords=coords, data_vars=data_vars)
    return nc


def make_fake_nc_file(file_name, shape, features):
    """Make nc file with dummy data for tests."""
    nc = make_fake_dset(shape, features)
    nc.to_netcdf(file_name)


class DummyData(AbstractContainer):
    """Dummy container with random data."""

    def __init__(self, data_shape, features):
        super().__init__()
        self.data = DataWrapper(make_fake_dset(data_shape, features))


class DummySampler(Sampler):
    """Dummy container with random data."""

    def __init__(self, sample_shape, data_shape, features, feature_sets=None):
        data = DummyData(data_shape=data_shape, features=features)
        super().__init__(data, sample_shape, feature_sets=feature_sets)


class DummyCroppedSampler(CroppedSampler):
    """Dummy container with random data."""

    def __init__(
        self,
        sample_shape,
        data_shape,
        features,
        feature_sets=None,
        crop_slice=slice(None),
    ):
        data = DummyData(data_shape=data_shape, features=features)
        super().__init__(
            data,
            sample_shape,
            feature_sets=feature_sets,
            crop_slice=crop_slice,
        )


def make_fake_h5_chunks(td):
    """Make fake h5 chunked output files for a 5x spatial 2x temporal
    multi-node forward pass output.

    Parameters
    ----------
    td : tempfile.TemporaryDirectory
        Test TemporaryDirectory

    Returns
    -------
    out_files : list
        List of filepaths to chunked files.
    data : ndarray
        (spatial_1, spatial_2, temporal, features)
        High resolution forward pass output
    ws_true : ndarray
        Windspeed between 0 and 20 in shape (spatial_1, spatial_2, temporal, 1)
    wd_true : ndarray
        Windir between 0 and 360 in shape (spatial_1, spatial_2, temporal, 1)
    features : list
        List of feature names corresponding to the last dimension of data
        ['windspeed_100m', 'winddirection_100m']
    t_slices_lr : list
        List of low res temporal slices
    t_slices_hr : list
        List of high res temporal slices
    s_slices_lr : list
        List of low res spatial slices
    s_slices_hr : list
        List of high res spatial slices
    low_res_lat_lon : ndarray
        Array of lat/lon for input data. (spatial_1, spatial_2, 2)
        Last dimension has ordering (lat, lon)
    low_res_times : list
        List of np.datetime64 objects for coarse data.
    """

    features = ['windspeed_100m', 'winddirection_100m']
    model_meta_data = {'foo': 'bar'}
    shape = (50, 50, 96, 1)
    ws_true = np.random.uniform(0, 20, shape)
    wd_true = np.random.uniform(0, 360, shape)
    data = np.concatenate((ws_true, wd_true), axis=3)
    lat = np.linspace(90, 0, 10)
    lon = np.linspace(-180, 0, 10)
    lon, lat = np.meshgrid(lon, lat)
    low_res_lat_lon = np.dstack((lat, lon))

    gids = np.arange(np.prod(shape[:2]))
    gids = gids.reshape(shape[:2])

    low_res_times = pd_date_range(
        '20220101', '20220103', freq='3600s', inclusive='left'
    )

    t_slices_lr = [slice(0, 24), slice(24, None)]
    t_slices_hr = [slice(0, 48), slice(48, None)]

    s_slices_lr = [slice(0, 5), slice(5, 10)]
    s_slices_hr = [slice(0, 25), slice(25, 50)]

    out_pattern = os.path.join(td, 'fp_out_{t}_{i}_{j}.h5')
    out_files = []
    for t, (slice_lr, slice_hr) in enumerate(zip(t_slices_lr, t_slices_hr)):
        for i, (s1_lr, s1_hr) in enumerate(zip(s_slices_lr, s_slices_hr)):
            for j, (s2_lr, s2_hr) in enumerate(zip(s_slices_lr, s_slices_hr)):
                out_file = out_pattern.format(
                    t=str(t).zfill(3),
                    i=str(i).zfill(3),
                    j=str(j).zfill(3),
                )
                out_files.append(out_file)
                OutputHandlerH5.write_output(
                    data[s1_hr, s2_hr, slice_hr, :],
                    features,
                    low_res_lat_lon[s1_lr, s2_lr],
                    low_res_times[slice_lr],
                    out_file,
                    meta_data=model_meta_data,
                    max_workers=1,
                    gids=gids[s1_hr, s2_hr],
                )

    return (
        out_files,
        data,
        ws_true,
        wd_true,
        features,
        t_slices_lr,
        t_slices_hr,
        s_slices_lr,
        s_slices_hr,
        low_res_lat_lon,
        low_res_times,
    )


def make_fake_cs_ratio_files(td, low_res_times, low_res_lat_lon, gan_meta):
    """Make a set of dummy clearsky ratio files that match the GAN fwp outputs

    Parameters
    ----------
    td : tempfile.TemporaryDirectory
        Test TemporaryDirectory
    low_res_times :
        List of times for low res input data. If there is only a single low
        res timestep, it is assumed the data is daily.
    low_res_lat_lon
        Array of lat/lon for input data.
        (spatial_1, spatial_2, 2)
        Last dimension has ordering (lat, lon)
    gan_meta : dict
        Meta data for model to write to file.

    Returns
    -------
    fps : list
        List of clearsky ratio .h5 chunked files.
    fp_pattern : str
        Glob pattern*string to find fps
    """
    fps = []
    chunk_dir = os.path.join(td, 'chunks/')
    fp_pattern = os.path.join(chunk_dir, 'sup3r_chunk_*.h5')
    os.makedirs(chunk_dir)

    for idt, timestamp in enumerate(low_res_times):
        fn = 'sup3r_chunk_{}_{}.h5'.format(str(idt).zfill(6), str(0).zfill(6))
        out_file = os.path.join(chunk_dir, fn)
        fps.append(out_file)

        cs_ratio = np.random.uniform(0, 1, (20, 20, 1, 1))
        cs_ratio = np.repeat(cs_ratio, 24, axis=2)

        OutputHandlerH5.write_output(
            cs_ratio,
            ['clearsky_ratio'],
            low_res_lat_lon,
            [timestamp],
            out_file,
            max_workers=1,
            meta_data=gan_meta,
        )
    return fps, fp_pattern
