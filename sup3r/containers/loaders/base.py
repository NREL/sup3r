"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging

import dask.array

from sup3r.containers.loaders.abstract import AbstractLoader

logger = logging.getLogger(__name__)


class Loader(AbstractLoader):
    """Base loader. "Loads" files so that a `.data` attribute provides access
    to the data in the files. This object provides a `__getitem__` method that
    can be used by Sampler objects to build batches or by Wrangler objects to
    derive / extract specific features / regions / time_periods."""

    DEFAULT_RES = None

    def __init__(
        self, file_paths, features, res_kwargs=None, chunks='auto', mode='lazy'
    ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list
            list of all features wanted from the file_paths.
        res_kwargs : dict
            kwargs for `.res` object
        chunks : tuple
            Tuple of chunk sizes to use for call to dask.array.from_array().
            Note: The ordering here corresponds to the default ordering given
            by `.res`.
        mode : str
            Options are ('lazy', 'eager') for how to load data.
        """
        super().__init__(
            file_paths=file_paths,
            features=features
        )
        self._res_kwargs = res_kwargs or {}
        self.mode = mode
        self.chunks = chunks
        self.data = self.load()

    @property
    def res(self):
        """Lowest level interface to data."""
        return self.DEFAULT_RES(self.file_paths, **self._res_kwargs)

    def load(self) -> dask.array:
        """Dask array with features in last dimension. Either lazily loaded
        (mode = 'lazy') or loaded into memory right away (mode = 'eager').

        Returns
        -------
        dask.array.core.Array
            (spatial, time, features) or (spatial_1, spatial_2, time, features)
        """
        data = dask.array.stack(
            [
                dask.array.from_array(self.res[f], chunks=self.chunks)
                for f in self.features
            ],
            axis=-1,
        )
        data = dask.array.moveaxis(data, 0, -2)

        if self.mode == 'eager':
            data = data.compute()

        return data

    def __getitem__(self, key):
        """Get data from container. This can be used to return a single sample
        from the underlying data for building batches or as part of extended
        feature extraction / derivation (spatial_1, spatial_2, temporal,
        features)."""
        out = self.data[key]
        if self.mode == 'lazy':
            out = out.compute(scheduler='threads')
        return out

    @property
    def shape(self):
        """Return shape of spatiotemporal extent available (spatial_1,
        spatial_2, temporal)"""
        return self.data.shape[:-1]
