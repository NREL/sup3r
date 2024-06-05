"""Base collection classes. These are objects that contain sets / lists of
containers like batch handlers. Of course these also contain data so they're
containers too!"""

from typing import List, Union

import numpy as np

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.dual import DualSampler


class Collection(Container):
    """Object consisting of a set of containers. These objects are distinct
    from :class:`Data` objects, which also contain multiple data members,
    because these members are completely independent of each other. They are
    collected together for the purpose of expanding a training dataset (e.g.
    BatchHandlers)."""

    def __init__(
        self,
        containers: Union[
            List[Container],
            List[Sampler],
            List[DualSampler],
        ],
    ):
        super().__init__()
        self.data = tuple(c.data for c in containers)
        self.containers = containers
        self._data_vars = []

    @property
    def data_vars(self):
        """Get all data vars contained in data."""
        if not self._data_vars:
            [
                self._data_vars.append(f)
                for f in np.concatenate([d.data_vars for d in self.data])
                if f not in self._data_vars
            ]
        return self._data_vars

    @property
    def container_weights(self):
        """Get weights used to sample from different containers based on
        relative sizes"""
        sizes = [c.size for c in self.containers]
        weights = sizes / np.sum(sizes)
        return weights.astype(np.float32)
