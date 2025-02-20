"""Slicer class for chunking forward pass input"""

import itertools as it
import logging
from dataclasses import dataclass
from typing import Optional, Union
from warnings import warn

import numpy as np

from sup3r.pipeline.utilities import (
    get_chunk_slices,
)
from sup3r.preprocessing.utilities import _parse_time_slice, log_args

logger = logging.getLogger(__name__)


@dataclass
class ForwardPassSlicer:
    """Get slices for sending data chunks through generator.

    Parameters
    ----------
    coarse_shape : tuple
        Shape of full domain for low res data
    time_steps : int
        Number of time steps for full temporal domain of low res data. This
        is used to construct a dummy_time_index from np.arange(time_steps)
    s_enhance : int
        Spatial enhancement factor
    t_enhance : int
        Temporal enhancement factor
    time_slice : slice | list
        Slice to use to extract range from time_index. Can be a ``slice(start,
        stop, step)`` or list ``[start, stop, step]``
    temporal_pad : int
        Size of temporal overlap between coarse chunks passed to forward
        passes for subsequent temporal stitching. This overlap will pad
        both sides of the fwp_chunk_shape. Note that the first and last
        chunks in the temporal dimension will not be padded.
    spatial_pad : int
        Size of spatial overlap between coarse chunks passed to forward
        passes for subsequent spatial stitching. This overlap will pad both
        sides of the fwp_chunk_shape. Note that the first and last chunks
        in any of the spatial dimension will not be padded.
    chunk_shape : tuple
        Max shape (spatial_1, spatial_2, temporal) of an unpadded coarse
        chunk to use for a forward pass. The number of nodes that the
        ForwardPassStrategy is set to distribute to is calculated by
        dividing up the total time index from all file_paths by the
        temporal part of this chunk shape. Each node will then be
        parallelized accross parallel processes by the spatial chunk shape.
        If temporal_pad / spatial_pad are non zero the chunk sent
        to the generator can be bigger than this shape. If running in
        serial set this equal to the shape of the full spatiotemporal data
        volume for best performance.
    min_width : tuple
        Minimum width of padded slices, with each element providing the min
        width for the corresponding dimension. e.g. (spatial_1, spatial_2,
        temporal). This is used to make sure generator network input meets the
        minimum size requirement for padding layers. e.g. If the generator
        includes a ``FlexiblePadding`` layer with ``padding = [0, 3, 3, 3, 0]``
        the minimum input shape to this layer must be ``[..., 4, 4, 4, ...]``
    """

    coarse_shape: Union[tuple, list]
    time_steps: int
    s_enhance: int
    t_enhance: int
    time_slice: slice
    temporal_pad: int
    spatial_pad: int
    chunk_shape: Union[tuple, list]
    min_width: Optional[Union[tuple, list]] = None

    @log_args
    def __post_init__(self):
        self.dummy_time_index = np.arange(self.time_steps)
        self.time_slice = _parse_time_slice(self.time_slice)

        self._chunk_lookup = None
        self._extra_padding = None
        self._s1_lr_slices = None
        self._s2_lr_slices = None
        self._s1_lr_pad_slices = None
        self._s2_lr_pad_slices = None
        self._s1_hr_crop_slices = None
        self._s2_hr_crop_slices = None
        self._s_lr_slices = None
        self._s_lr_pad_slices = None
        self._s_lr_crop_slices = None
        self._t_lr_pad_slices = None
        self._t_lr_crop_slices = None
        self._s_hr_slices = None
        self._s_hr_crop_slices = None
        self._t_hr_crop_slices = None
        self._hr_crop_slices = None
        self.min_width = (
            self.chunk_shape if self.min_width is None else self.min_width
        )

    def get_spatial_slices(self):
        """Get spatial slices for small data chunks that are passed through
        generator

        Returns
        -------
        s_lr_slices: list
            List of slices for low res data chunks which have not been padded.
            data_handler.data[s_lr_slice] corresponds to an unpadded low res
            input to the model.
        s_lr_pad_slices : list
            List of slices which have been padded so that high res output
            can be stitched together. data_handler.data[s_lr_pad_slice]
            corresponds to a padded low res input to the model.
        s_hr_slices : list
            List of slices for high res data corresponding to the
            lr_slices regions. output_array[s_hr_slice] corresponds to the
            cropped generator output.
        """
        return (self.s_lr_slices, self.s_lr_pad_slices, self.s_hr_slices)

    def get_time_slices(self):
        """Calculate the number of time chunks across the full time index

        Returns
        -------
        t_lr_slices : list
            List of low-res non-padded time index slices. e.g. If
            fwp_chunk_size[2] is 5 then the size of these slices will always
            be 5.
        t_lr_pad_slices : list
            List of low-res padded time index slices. e.g. If fwp_chunk_size[2]
            is 5 the size of these slices will be 15, with exceptions at the
            start and end of the full time index.
        """
        return self.t_lr_slices, self.t_lr_pad_slices

    @property
    def s_lr_slices(self):
        """Get low res spatial slices for small data chunks that are passed
        through generator

        Returns
        -------
        _s_lr_slices : list
            List of spatial slices corresponding to the unpadded spatial region
            going through the generator
        """
        if self._s_lr_slices is None:
            self._s_lr_slices = list(
                it.product(self.s1_lr_slices, self.s2_lr_slices)
            )
        return self._s_lr_slices

    @property
    def s_lr_pad_slices(self):
        """Get low res padded slices for small data chunks that are passed
        through generator

        Returns
        -------
        _s_lr_pad_slices : list
            List of slices which have been padded so that high res output
            can be stitched together. Each entry in this list has a slice for
            each spatial dimension. data_handler.data[s_lr_pad_slice] gives the
            padded data volume passed through the generator
        """
        if self._s_lr_pad_slices is None:
            self._s_lr_pad_slices = list(
                it.product(self.s1_lr_pad_slices, self.s2_lr_pad_slices)
            )
        return self._s_lr_pad_slices

    @property
    def t_lr_pad_slices(self):
        """Get low res temporal padded slices for distributing time chunks
        across nodes. These slices correspond to the time chunks sent to each
        node and are padded according to temporal_pad.

        Returns
        -------
        _t_lr_pad_slices : list
            List of low res temporal slices which have been padded so that high
            res output can be stitched together
        """
        if self._t_lr_pad_slices is None:
            self._t_lr_pad_slices = self.get_padded_slices(
                slices=self.t_lr_slices,
                shape=self.time_steps,
                enhancement=1,
                padding=self.temporal_pad,
                step=self.time_slice.step,
            )
        return self._t_lr_pad_slices

    @property
    def t_lr_crop_slices(self):
        """Get low res temporal cropped slices for cropping time index of
        padded input data.

        Returns
        -------
        _t_lr_crop_slices : list
            List of low res temporal slices for cropping padded input data
        """
        if self._t_lr_crop_slices is None:
            self._t_lr_crop_slices = self.get_cropped_slices(
                self.t_lr_slices, self.t_lr_pad_slices, 1
            )

        return self._t_lr_crop_slices

    @property
    def t_hr_crop_slices(self):
        """Get high res temporal cropped slices for cropping forward pass
        output before stitching together

        Returns
        -------
        _t_hr_crop_slices : list
            List of high res temporal slices for cropping padded generator
            output
        """
        hr_crop_start = None
        hr_crop_stop = None
        if self.temporal_pad > 0:
            hr_crop_start = self.t_enhance * self.temporal_pad
            hr_crop_stop = -hr_crop_start

        if self._t_hr_crop_slices is None:
            # don't use self.get_cropped_slices() here because temporal padding
            # gets weird at beginning and end of timeseries and the temporal
            # axis should always be evenly chunked.
            self._t_hr_crop_slices = [
                slice(hr_crop_start, hr_crop_stop)
                for _ in range(len(self.t_lr_slices))
            ]

        return self._t_hr_crop_slices

    @property
    def s1_hr_slices(self):
        """Get high res spatial slices for first spatial dimension"""
        return self.get_hr_slices(self.s1_lr_slices, self.s_enhance)

    @property
    def s2_hr_slices(self):
        """Get high res spatial slices for second spatial dimension"""
        return self.get_hr_slices(self.s2_lr_slices, self.s_enhance)

    @property
    def s1_hr_crop_slices(self):
        """Get high res cropped slices for first spatial dimension"""

        if self._s1_hr_crop_slices is None:
            hr_crop_start = self.s_enhance * self.spatial_pad or None
            hr_crop_stop = None if self.spatial_pad == 0 else -hr_crop_start

            self._s1_hr_crop_slices = [
                slice(hr_crop_start, hr_crop_stop)
            ] * len(self.s1_lr_slices)

            self._s1_hr_crop_slices = self.check_boundary_slice(
                unpadded_slices=self.s1_lr_slices,
                cropped_slices=self._s1_hr_crop_slices,
                enhancement=self.s_enhance,
                padding=self.spatial_pad,
                dim=0,
            )
        return self._s1_hr_crop_slices

    @property
    def s2_hr_crop_slices(self):
        """Get high res cropped slices for first spatial dimension"""

        if self._s2_hr_crop_slices is None:
            hr_crop_start = self.s_enhance * self.spatial_pad or None
            hr_crop_stop = None if self.spatial_pad == 0 else -hr_crop_start

            self._s2_hr_crop_slices = [
                slice(hr_crop_start, hr_crop_stop)
            ] * len(self.s2_lr_slices)

            self._s2_hr_crop_slices = self.check_boundary_slice(
                unpadded_slices=self.s2_lr_slices,
                cropped_slices=self._s2_hr_crop_slices,
                enhancement=self.s_enhance,
                padding=self.spatial_pad,
                dim=1,
            )
        return self._s2_hr_crop_slices

    @property
    def s_hr_slices(self):
        """Get high res slices for indexing full generator output array

        Returns
        -------
        _s_hr_slices : list
            List of high res slices. Each entry in this list has a slice for
            each spatial dimension. output[hr_slice] gives the superresolved
            domain corresponding to data_handler.data[lr_slice]
        """
        if self._s_hr_slices is None:
            self._s_hr_slices = list(
                it.product(self.s1_hr_slices, self.s2_hr_slices)
            )
        return self._s_hr_slices

    @property
    def s_lr_crop_slices(self):
        """Get low res cropped slices for cropping input chunk domain

        Returns
        -------
        _s_lr_crop_slices : list
            List of low res cropped slices. Each entry in this list has a
            slice for each spatial dimension.
        """
        if self._s_lr_crop_slices is None:
            self._s_lr_crop_slices = []
            s1_crop_slices = self.get_cropped_slices(
                self.s1_lr_slices, self.s1_lr_pad_slices, 1
            )

            s1_crop_slices = self.check_boundary_slice(
                unpadded_slices=self.s1_lr_slices,
                cropped_slices=s1_crop_slices,
                enhancement=self.s_enhance,
                padding=self.spatial_pad,
                dim=0,
            )
            s2_crop_slices = self.get_cropped_slices(
                self.s2_lr_slices, self.s2_lr_pad_slices, 1
            )
            s2_crop_slices = self.check_boundary_slice(
                unpadded_slices=self.s2_lr_slices,
                cropped_slices=s2_crop_slices,
                enhancement=self.s_enhance,
                padding=self.spatial_pad,
                dim=1,
            )
            self._s_lr_crop_slices = list(
                it.product(s1_crop_slices, s2_crop_slices)
            )
        return self._s_lr_crop_slices

    @property
    def s_hr_crop_slices(self):
        """Get high res cropped slices for cropping generator output

        Returns
        -------
        _s_hr_crop_slices : list
            List of high res cropped slices. Each entry in this list has a
            slice for each spatial dimension.
        """
        if self._s_hr_crop_slices is None:
            self._s_hr_crop_slices = list(
                it.product(self.s1_hr_crop_slices, self.s2_hr_crop_slices)
            )
        return self._s_hr_crop_slices

    @property
    def hr_crop_slices(self):
        """Get high res spatiotemporal cropped slices for cropping generator
        output

        Returns
        -------
        _hr_crop_slices : list
            List of high res spatiotemporal cropped slices. Each entry in this
            list has a crop slice for each spatial dimension and temporal
            dimension and then slice(None) for the feature dimension.
            model.generate()[hr_crop_slice] gives the cropped generator output
            corresponding to outpuUnion[np.ndarray, da.core.Array][hr_slice]
        """
        if self._hr_crop_slices is None:
            self._hr_crop_slices = []
            for t in self.t_hr_crop_slices:
                node_slices = [
                    (s[0], s[1], t, slice(None)) for s in self.s_hr_crop_slices
                ]
                self._hr_crop_slices.append(node_slices)
        return self._hr_crop_slices

    @property
    def s1_lr_pad_slices(self):
        """List of low resolution spatial slices with padding for first
        spatial dimension"""
        if self._s1_lr_pad_slices is None:
            self._s1_lr_pad_slices = self.get_padded_slices(
                slices=self.s1_lr_slices,
                shape=self.coarse_shape[0],
                enhancement=1,
                padding=self.spatial_pad,
            )
        return self._s1_lr_pad_slices

    @property
    def s2_lr_pad_slices(self):
        """List of low resolution spatial slices with padding for second
        spatial dimension"""
        if self._s2_lr_pad_slices is None:
            self._s2_lr_pad_slices = self.get_padded_slices(
                slices=self.s2_lr_slices,
                shape=self.coarse_shape[1],
                enhancement=1,
                padding=self.spatial_pad,
            )
        return self._s2_lr_pad_slices

    @property
    def s1_lr_slices(self):
        """List of low resolution spatial slices for first spatial dimension
        considering padding on all sides of the spatial raster."""
        ind = slice(0, self.coarse_shape[0])
        return get_chunk_slices(
            self.coarse_shape[0], self.chunk_shape[0], index_slice=ind
        )

    @property
    def s2_lr_slices(self):
        """List of low resolution spatial slices for second spatial dimension
        considering padding on all sides of the spatial raster."""
        ind = slice(0, self.coarse_shape[1])
        return get_chunk_slices(
            self.coarse_shape[1], self.chunk_shape[1], index_slice=ind
        )

    @property
    def t_lr_slices(self):
        """Low resolution temporal slices"""
        n_tsteps = len(self.dummy_time_index[self.time_slice])
        n_chunks = n_tsteps / self.chunk_shape[2]
        n_chunks = int(np.ceil(n_chunks))
        ti_slices = self.dummy_time_index[self.time_slice]
        ti_slices = np.array_split(ti_slices, n_chunks)
        return [
            slice(c[0], c[-1] + 1, self.time_slice.step) for c in ti_slices
        ]

    @staticmethod
    def get_hr_slices(slices, enhancement, step=None):
        """Get high resolution slices for temporal or spatial slices

        Parameters
        ----------
        slices : list
            Low resolution slices to be enhanced
        enhancement : int
            Enhancement factor
        step : int | None
            Step size for slices

        Returns
        -------
        hr_slices : list
            High resolution slices
        """
        hr_slices = []
        if step is not None:
            step *= enhancement
        for sli in slices:
            start = sli.start * enhancement
            stop = sli.stop * enhancement
            hr_slices.append(slice(start, stop, step))
        return hr_slices

    @property
    def chunk_lookup(self):
        """Get a 3D array with shape
        (n_spatial_1_chunks, n_spatial_2_chunks, n_time_chunks)
        where each value is the chunk index."""
        if self._chunk_lookup is None:
            n_s1 = len(self.s1_lr_slices)
            n_s2 = len(self.s2_lr_slices)
            n_t = self.n_time_chunks
            lookup = np.arange(self.n_chunks).reshape((n_t, n_s1, n_s2))
            self._chunk_lookup = np.transpose(lookup, axes=(1, 2, 0))
        return self._chunk_lookup

    @property
    def spatial_chunk_lookup(self):
        """Get a 2D array with shape (n_spatial_1_chunks, n_spatial_2_chunks)
        where each value is the spatial chunk index."""
        n_s1 = len(self.s1_lr_slices)
        n_s2 = len(self.s2_lr_slices)
        return np.arange(self.n_spatial_chunks).reshape((n_s1, n_s2))

    @property
    def n_spatial_chunks(self):
        """Get the number of spatial chunks"""
        return len(self.hr_crop_slices[0])

    @property
    def n_time_chunks(self):
        """Get the number of temporal chunks"""
        return len(self.t_hr_crop_slices)

    @property
    def n_chunks(self):
        """Get total number of spatiotemporal chunks"""
        return self.n_spatial_chunks * self.n_time_chunks

    @staticmethod
    def get_padded_slices(slices, shape, enhancement, padding, step=None):
        """Get padded slices with the specified padding size, max shape,
        enhancement, and step size

        Parameters
        ----------
        slices : list
            List of low res unpadded slice
        shape : int
            max possible index of a padded slice. e.g. if the slices are
            indexing a dimension with size 10 then a padded slice cannot have
            an index greater than 10.
        enhancement : int
            Enhancement factor. e.g. If these slices are indexing a spatial
            dimension which will be enhanced by 2x then enhancement=2.
        padding : int
            Padding factor. e.g. If these slices are indexing a spatial
            dimension and the spatial_pad is 10 this is 10. It will be
            multiplied by the enhancement factor if the slices are to be used
            to index an enhanced dimension.
        step : int | None
            Step size for slices. e.g. If these slices are indexing a temporal
            dimension and time_slice.step = 3 then step=3.

        Returns
        -------
        list
            Padded slices for temporal or spatial dimensions.
        """
        step = step or 1
        pad = step * padding * enhancement
        pad_slices = []
        for _, s in enumerate(slices):
            end = np.min([enhancement * shape, s.stop * enhancement + pad])
            start = np.max([0, s.start * enhancement - pad])
            pad_slices.append(slice(start, end, step))
        return pad_slices

    def check_boundary_slice(
        self, unpadded_slices, cropped_slices, enhancement, padding, dim
    ):
        """Check cropped slice at the right boundary for minimum shape.

        It is possible for the forward pass chunk shape to divide the grid size
        such that the last slice (right boundary) does not meet the minimum
        number of elements. (Padding layers in the generator require a minimum
        shape). e.g. ``grid_size = (8, 8)`` with ``fwp_chunk_shape = (7, 7,
        ...)`` results in unpadded slices with just one element. When this
        minimum shape is not met we apply extra padding in
        :meth:`self._get_pad_width`. Cropped slices have to be adjusted to
        account for this here."""

        warn_msg = (
            'The final slice for dimension #%s is too small '
            '(slice=slice(%s, %s), padding=%s). The start of this slice will '
            'be reduced to try to meet the minimum slice length.'
        )

        lr_slice_start = unpadded_slices[-1].start or 0
        lr_slice_stop = unpadded_slices[-1].stop or self.coarse_shape[dim]

        # last slice adjustment
        padded_width = 2 * padding + lr_slice_stop - lr_slice_start
        too_small = padded_width < self.min_width[dim]
        if too_small:
            half_width = self.min_width[dim] // 2 + 1
            logger.warning(
                warn_msg,
                dim + 1,
                lr_slice_start,
                lr_slice_stop,
                padding,
            )
            warn(warn_msg % (dim + 1, lr_slice_start, lr_slice_stop, padding))
            cropped_slices[-1] = slice(
                half_width * enhancement, -half_width * enhancement
            )

        return cropped_slices

    @staticmethod
    def get_cropped_slices(unpadded_slices, padded_slices, enhancement):
        """Get cropped slices to cut off padded output

        Parameters
        ----------
        unpadded_slices : list
            List of unpadded slices
        padded_slices : list
            List of padded slices
        enhancement : int
            Enhancement factor for the data to be cropped.

        Returns
        -------
        list
            Cropped slices for temporal or spatial dimensions.
        """
        cropped_slices = []
        for ps, us in zip(padded_slices, unpadded_slices):
            start = us.start
            stop = us.stop
            step = us.step or 1
            if start is not None:
                start = enhancement * (us.start - ps.start) // step
            if stop is not None:
                stop = enhancement * (us.stop - ps.stop) // step
            if start is not None and start <= 0:
                start = None
            if stop is not None and stop >= 0:
                stop = None
            cropped_slices.append(slice(start, stop))

        return cropped_slices

    @staticmethod
    def _get_pad_width(
        window, max_steps, max_pad, min_width=None, check_boundary=False
    ):
        """
        Parameters
        ----------
        window : slice
            Slice with start and stop of window to pad.
        max_steps : int
            Maximum number of steps available. Padding cannot extend past this
        max_pad : int
            Maximum amount of padding to apply.
        min_width : int | None
            Minimum width to enforce. This could be the forward pass chunk
            shape or 1 + padding value in the first padding layer of the
            generator network. This is only used if ``check_boundary = True``
        check_bounary : bool
            Whether to check the final slice for minimum size requirement

        Returns
        -------
        tuple
            Tuple of pad width for the given window.
        """
        win_start = window.start or 0
        win_stop = window.stop or max_steps
        start = int(np.maximum(0, (max_pad - win_start)))
        stop = int(np.maximum(0, max_pad + win_stop - max_steps))

        # We add minimum padding to the last slice if the padded window is
        # too small for the generator. This can happen if 2 * spatial_pad +
        # modulo(grid_size, fwp_chunk_shape) is less than the padding applied
        # in the first padding layer of the generator
        padded_width = 2 * max_pad + win_stop - win_start
        is_last_slice = win_stop == max_steps
        too_small = min_width is not None and padded_width < min_width
        if check_boundary and is_last_slice and too_small:
            half_width = min_width // 2 + 1
            stop = np.max([half_width, max_pad])
            start = np.max([half_width, max_pad])

        return (start, stop)

    def get_chunk_indices(self, chunk_index):
        """Get (spatial, temporal) indices for the given chunk index"""
        return (
            chunk_index % self.n_spatial_chunks,
            chunk_index // self.n_spatial_chunks,
        )

    def get_pad_width(self, chunk_index):
        """Get extra padding for the current spatiotemporal chunk

        Returns
        -------
        padding : tuple
            Tuple of tuples with padding width for spatial and temporal
            dimensions. Each tuple includes the start and end of padding for
            that dimension. Ordering is spatial_1, spatial_2, temporal.
        """
        s_chunk_idx, t_chunk_idx = self.get_chunk_indices(chunk_index)
        ti_slice = self.t_lr_slices[t_chunk_idx]
        lr_slice = self.s_lr_slices[s_chunk_idx]

        return (
            self._get_pad_width(
                lr_slice[0],
                self.coarse_shape[0],
                self.spatial_pad,
                self.min_width[0],
                check_boundary=True,
            ),
            self._get_pad_width(
                lr_slice[1],
                self.coarse_shape[1],
                self.spatial_pad,
                self.min_width[1],
                check_boundary=True,
            ),
            self._get_pad_width(
                ti_slice, len(self.dummy_time_index), self.temporal_pad
            ),
        )

    @property
    def extra_padding(self):
        """Get list of pad widths for each chunk index"""
        if self._extra_padding is None:
            self._extra_padding = [
                self.get_pad_width(idx) for idx in range(self.n_chunks)
            ]
        return self._extra_padding
