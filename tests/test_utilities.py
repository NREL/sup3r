# -*- coding: utf-8 -*-
"""pytests for data handling"""

import numpy as np

from sup3r.utilities.utilities import (get_chunk_slices,
                                       weighted_time_sampler,
                                       uniform_time_sampler)


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
    chunks = np.array_split(np.arange(0, 100), 10)
    weights = np.zeros(10)
    weights_1 = weights.copy()
    weights_1[0] = 1

    weights_2 = weights.copy()
    weights_2[-1] = 1

    weights_3 = weights.copy()
    weights_3[2] = 0.5
    weights_3[5] = 0.5

    for _ in range(100):

        slice_1 = weighted_time_sampler(data, 10, weights_1)
        assert 0 <= slice_1.start <= 10

        slice_2 = weighted_time_sampler(data, 10, weights_2)
        assert 90 <= slice_2.start <= 100

        slice_3 = weighted_time_sampler(data, 10, weights_3)
        assert (chunks[2][0] <= slice_3.start <= chunks[2][-1]
               or chunks[5][0] <= slice_3.start <= chunks[5][-1])


def test_uniform_time_sampler():
    """Test uniform_time_sampler for correct start point based on
    temporal_focus"""

    data = np.zeros((1, 1, 100))
    temporal_focus = slice(10, 30)

    for _ in range(100):

        slice_1 = uniform_time_sampler(data, 10, temporal_focus)
        assert 10 <= slice_1.start <= 30
