"""Slicer class for chunking forward pass input"""

import logging

import numpy as np

from sup3r.utilities.utilities import (
    get_chunk_slices,
)

logger = logging.getLogger(__name__)


class ForwardPassSlicer:
    """Get slices for sending data chunks through generator."""

    def __init__(
        self,
        coarse_shape,
        time_steps,
        time_slice,
        chunk_shape,
        s_enhancements,
        t_enhancements,
        spatial_pad,
        temporal_pad,
    ):
        """
        Parameters
        ----------
        coarse_shape : tuple
            Shape of full domain for low res data
        time_steps : int
            Number of time steps for full temporal domain of low res data. This
            is used to construct a dummy_time_index from np.arange(time_steps)
        time_slice : slice
            Slice to use to extract range from time_index
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
        s_enhancements : list
            List of factors by which the Sup3rGan model will enhance the
            spatial dimensions of low resolution data. If there are two 5x
            spatial enhancements, this should be [5, 5] where the total
            enhancement is the product of these factors.
        t_enhancements : list
            List of factor by which the Sup3rGan model will enhance temporal
            dimension of low resolution data
        spatial_pad : int
            Size of spatial overlap between coarse chunks passed to forward
            passes for subsequent spatial stitching. This overlap will pad both
            sides of the fwp_chunk_shape. Note that the first and last chunks
            in any of the spatial dimension will not be padded.
        temporal_pad : int
            Size of temporal overlap between coarse chunks passed to forward
            passes for subsequent temporal stitching. This overlap will pad
            both sides of the fwp_chunk_shape. Note that the first and last
            chunks in the temporal dimension will not be padded.
        """
        self.grid_shape = coarse_shape
        self.time_steps = time_steps
        self.s_enhancements = s_enhancements
        self.t_enhancements = t_enhancements
        self.s_enhance = np.prod(self.s_enhancements)
        self.t_enhance = np.prod(self.t_enhancements)
        self.dummy_time_index = np.arange(time_steps)
        self.time_slice = time_slice
        self.temporal_pad = temporal_pad
        self.spatial_pad = spatial_pad
        self.chunk_shape = chunk_shape

        self._chunk_lookup = None
        self._s1_lr_slices = None
        self._s2_lr_slices = None
        self._s1_lr_pad_slices = None
        self._s2_lr_pad_slices = None
        self._s_lr_slices = None
        self._s_lr_pad_slices = None
        self._s_lr_crop_slices = None
        self._t_lr_pad_slices = None
        self._t_lr_crop_slices = None
        self._s_hr_slices = None
        self._s_hr_crop_slices = None
        self._t_hr_crop_slices = None
        self._hr_crop_slices = None

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
            self._s_lr_slices = []
            for _, s1 in enumerate(self.s1_lr_slices):
                for _, s2 in enumerate(self.s2_lr_slices):
                    s_slice = (s1, s2, slice(None), slice(None))
                    self._s_lr_slices.append(s_slice)
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
            each spatial dimension and then slice(None) for temporal and
            feature dimension. This is because the temporal dimension is only
            chunked across nodes and not within a single node.
            data_handler.data[s_lr_pad_slice] gives the padded data volume
            passed through the generator
        """
        if self._s_lr_pad_slices is None:
            self._s_lr_pad_slices = []
            for _, s1 in enumerate(self.s1_lr_pad_slices):
                for _, s2 in enumerate(self.s2_lr_pad_slices):
                    pad_slice = (s1, s2, slice(None), slice(None))
                    self._s_lr_pad_slices.append(pad_slice)

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
                self.t_lr_slices,
                self.time_steps,
                1,
                self.temporal_pad,
                self.time_slice.step,
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
    def s_hr_slices(self):
        """Get high res slices for indexing full generator output array

        Returns
        -------
        _s_hr_slices : list
            List of high res slices. Each entry in this list has a slice for
            each spatial dimension and then slice(None) for temporal and
            feature dimension. This is because the temporal dimension is only
            chunked across nodes and not within a single node. output[hr_slice]
            gives the superresolved domain corresponding to
            data_handler.data[lr_slice]
        """
        if self._s_hr_slices is None:
            self._s_hr_slices = []
            for _, s1 in enumerate(self.s1_hr_slices):
                for _, s2 in enumerate(self.s2_hr_slices):
                    hr_slice = (s1, s2, slice(None), slice(None))
                    self._s_hr_slices.append(hr_slice)
        return self._s_hr_slices

    @property
    def s_lr_crop_slices(self):
        """Get low res cropped slices for cropping input chunk domain

        Returns
        -------
        _s_lr_crop_slices : list
            List of low res cropped slices. Each entry in this list has a
            slice for each spatial dimension and then slice(None) for temporal
            and feature dimension.
        """
        if self._s_lr_crop_slices is None:
            self._s_lr_crop_slices = []
            s1_crop_slices = self.get_cropped_slices(
                self.s1_lr_slices, self.s1_lr_pad_slices, 1
            )
            s2_crop_slices = self.get_cropped_slices(
                self.s2_lr_slices, self.s2_lr_pad_slices, 1
            )
            for i, _ in enumerate(self.s1_lr_slices):
                for j, _ in enumerate(self.s2_lr_slices):
                    lr_crop_slice = (
                        s1_crop_slices[i],
                        s2_crop_slices[j],
                        slice(None),
                        slice(None),
                    )
                    self._s_lr_crop_slices.append(lr_crop_slice)
        return self._s_lr_crop_slices

    @property
    def s_hr_crop_slices(self):
        """Get high res cropped slices for cropping generator output

        Returns
        -------
        _s_hr_crop_slices : list
            List of high res cropped slices. Each entry in this list has a
            slice for each spatial dimension and then slice(None) for temporal
            and feature dimension.
        """
        hr_crop_start = None
        hr_crop_stop = None
        if self.spatial_pad > 0:
            hr_crop_start = self.s_enhance * self.spatial_pad
            hr_crop_stop = -hr_crop_start

        if self._s_hr_crop_slices is None:
            self._s_hr_crop_slices = []
            s1_hr_crop_slices = [
                slice(hr_crop_start, hr_crop_stop)
                for _ in range(len(self.s1_lr_slices))
            ]
            s2_hr_crop_slices = [
                slice(hr_crop_start, hr_crop_stop)
                for _ in range(len(self.s2_lr_slices))
            ]

            for _, s1 in enumerate(s1_hr_crop_slices):
                for _, s2 in enumerate(s2_hr_crop_slices):
                    hr_crop_slice = (s1, s2, slice(None), slice(None))
                    self._s_hr_crop_slices.append(hr_crop_slice)
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
            corresponding to output_array[hr_slice]
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
                self.s1_lr_slices,
                self.grid_shape[0],
                1,
                padding=self.spatial_pad,
            )
        return self._s1_lr_pad_slices

    @property
    def s2_lr_pad_slices(self):
        """List of low resolution spatial slices with padding for second
        spatial dimension"""
        if self._s2_lr_pad_slices is None:
            self._s2_lr_pad_slices = self.get_padded_slices(
                self.s2_lr_slices,
                self.grid_shape[1],
                1,
                padding=self.spatial_pad,
            )
        return self._s2_lr_pad_slices

    @property
    def s1_lr_slices(self):
        """List of low resolution spatial slices for first spatial dimension
        considering padding on all sides of the spatial raster."""
        ind = slice(0, self.grid_shape[0])
        slices = get_chunk_slices(
            self.grid_shape[0], self.chunk_shape[0], index_slice=ind
        )
        return slices

    @property
    def s2_lr_slices(self):
        """List of low resolution spatial slices for second spatial dimension
        considering padding on all sides of the spatial raster."""
        ind = slice(0, self.grid_shape[1])
        slices = get_chunk_slices(
            self.grid_shape[1], self.chunk_shape[1], index_slice=ind
        )
        return slices

    @property
    def t_lr_slices(self):
        """Low resolution temporal slices"""
        n_tsteps = len(self.dummy_time_index[self.time_slice])
        n_chunks = n_tsteps / self.chunk_shape[2]
        n_chunks = int(np.ceil(n_chunks))
        ti_slices = self.dummy_time_index[self.time_slice]
        ti_slices = np.array_split(ti_slices, n_chunks)
        ti_slices = [
            slice(c[0], c[-1] + 1, self.time_slice.step) for c in ti_slices
        ]
        return ti_slices

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
            start = np.max([0, s.start * enhancement - pad])
            end = np.min([enhancement * shape, s.stop * enhancement + pad])
            pad_slices.append(slice(start, end, step))
        return pad_slices

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