# -*- coding: utf-8 -*-
"""pytests for general utilities"""
import numpy as np
import pytest

from sup3r.utilities.utilities import (get_chunk_slices,
                                       uniform_time_sampler,
                                       weighted_time_sampler,
                                       uniform_box_sampler,
                                       spatial_coarsening)


def test_get_chunk_slices():
    """Test get_chunk_slices function for correct start/end"""

    arr = np.arange(10, 100)
    index_slice = slice(20, 80)

    slices = get_chunk_slices(len(arr), chunk_size=10, index_slice=index_slice)

    assert slices[0].start == index_slice.start
    assert slices[-1].stop == index_slice.stop


def test_weighted_time_sampler():
    """Test weighted_time_sampler for correct start point based on weights"""

    data = np.zeros((1, 1, 100))
    shape = 10
    chunks = np.array_split(np.arange(0, 100 - (shape - 1)), 10)
    weights = np.zeros(len(chunks))
    weights_1 = weights.copy()
    weights_1[0] = 1

    weights_2 = weights.copy()
    weights_2[-1] = 1

    weights_3 = weights.copy()
    weights_3[2] = 0.5
    weights_3[5] = 0.5

    for _ in range(100):

        slice_1 = weighted_time_sampler(data, shape, weights_1)
        assert chunks[0][0] <= slice_1.start <= chunks[0][-1]

        slice_2 = weighted_time_sampler(data, shape, weights_2)
        assert chunks[-1][0] <= slice_2.start <= chunks[-1][-1]

        slice_3 = weighted_time_sampler(data, 10, weights_3)
        assert (chunks[2][0] <= slice_3.start <= chunks[2][-1]
               or chunks[5][0] <= slice_3.start <= chunks[5][-1])

    shape = 1
    weights = np.zeros(data.shape[2])
    weights_4 = weights.copy()
    weights_4[5] = 1

    slice_4 = weighted_time_sampler(data, shape, weights_4)
    print(slice_4)
    assert weights_4[slice_4.start] == 1


def test_uniform_time_sampler():
    """Test uniform_time_sampler for correct start point for edge case"""

    data = np.zeros((1, 1, 10))
    shape = 10
    t_slice = uniform_time_sampler(data, shape)
    assert t_slice.start == 0
    assert t_slice.stop == data.shape[2]


def test_uniform_box_sampler():
    """Test uniform_box_sampler for correct start point for edge case"""

    data = np.zeros((10, 10, 1))
    shape = (10, 10)
    [s1, s2] = uniform_box_sampler(data, shape)
    assert s1.start == s2.start == 0
    assert s1.stop == data.shape[0]
    assert s2.stop == data.shape[1]


def test_s_enhance_errors():
    """Negative tests of spatial coarsening method"""

    arr = np.arange(28800).reshape((2, 20, 20, 12, 3))
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=3)
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=7)
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=40)

    arr = np.ones(10)
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=5)

    arr = np.ones((4, 4))
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=2)


@pytest.mark.parametrize('s_enhance', [1, 2, 4, 5])
def test_s_enhance_5D(s_enhance):
    """Test the spatial enhancement of a 5D array"""
    arr = np.arange(28800).reshape((2, 20, 20, 12, 3))
    coarse = spatial_coarsening(arr, s_enhance=s_enhance, obs_axis=True)

    for o in range(arr.shape[0]):
        for t in range(arr.shape[3]):
            for f in range(arr.shape[4]):
                for i_lr in range(coarse.shape[1]):
                    for j_lr in range(coarse.shape[2]):

                        i_hr = i_lr * s_enhance
                        i_hr = slice(i_hr, i_hr + s_enhance)

                        j_hr = j_lr * s_enhance
                        j_hr = slice(j_hr, j_hr + s_enhance)

                        assert np.allclose(coarse[o, i_lr, j_lr, t, f],
                                           arr[o, i_hr, j_hr, t, f].mean())


@pytest.mark.parametrize('s_enhance', [1, 2, 4, 5])
def test_s_enhance_4D(s_enhance):
    """Test the spatial enhancement of a 4D array"""
    arr = np.arange(28800).reshape((24, 20, 20, 3))
    coarse = spatial_coarsening(arr, s_enhance=s_enhance, obs_axis=True)

    for o in range(arr.shape[0]):
        for f in range(arr.shape[3]):
            for i_lr in range(coarse.shape[1]):
                for j_lr in range(coarse.shape[2]):

                    i_hr = i_lr * s_enhance
                    i_hr = slice(i_hr, i_hr + s_enhance)

                    j_hr = j_lr * s_enhance
                    j_hr = slice(j_hr, j_hr + s_enhance)

                    assert np.allclose(coarse[o, i_lr, j_lr, f],
                                       arr[o, i_hr, j_hr, f].mean())


@pytest.mark.parametrize('s_enhance', [1, 2, 4, 5])
def test_s_enhance_4D_no_obs(s_enhance):
    """Test the spatial enhancement of a 4D array without the obs axis"""
    arr = np.arange(28800).reshape((20, 20, 24, 3))
    coarse = spatial_coarsening(arr, s_enhance=s_enhance, obs_axis=False)

    for t in range(arr.shape[2]):
        for f in range(arr.shape[3]):
            for i_lr in range(coarse.shape[0]):
                for j_lr in range(coarse.shape[1]):

                    i_hr = i_lr * s_enhance
                    i_hr = slice(i_hr, i_hr + s_enhance)

                    j_hr = j_lr * s_enhance
                    j_hr = slice(j_hr, j_hr + s_enhance)

                    assert np.allclose(coarse[i_lr, j_lr, t, f],
                                       arr[i_hr, j_hr, t, f].mean())


@pytest.mark.parametrize('s_enhance', [1, 2, 4, 5])
def test_s_enhance_3D_no_obs(s_enhance):
    """Test the spatial enhancement of a 3D array without the obs axis"""
    arr = np.arange(28800).reshape((20, 20, 72))
    coarse = spatial_coarsening(arr, s_enhance=s_enhance, obs_axis=False)

    for f in range(arr.shape[2]):
        for i_lr in range(coarse.shape[0]):
            for j_lr in range(coarse.shape[1]):

                i_hr = i_lr * s_enhance
                i_hr = slice(i_hr, i_hr + s_enhance)

                j_hr = j_lr * s_enhance
                j_hr = slice(j_hr, j_hr + s_enhance)

                assert np.allclose(coarse[i_lr, j_lr, f],
                                   arr[i_hr, j_hr, f].mean())
