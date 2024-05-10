from abc import ABC, abstractmethod

from sup3r.containers.loaders.abstract import AbstractLoader


class AbstractWrangler(AbstractLoader, ABC):
    """Loader subclass with additional methods for wrangling data. e.g.
    Extracting specific spatiotemporal extents and features and deriving new
    features."""

    @abstractmethod
    def get_raster_index(self):
        """Get array of indices used to select the spatial region of
        interest."""

    @abstractmethod
    def get_time_index(self):
        """Get the time index for the time period of interest."""
