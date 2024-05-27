"""Top level containers. These are just things that have access to data.
Loaders, Extracters, Samplers, Derivers, Handlers, Batchers, etc are subclasses
of Containers. Rather than having a single object that does everything -
extract data, compute features, sample the data for batching, split into train
and val, etc, we have fundamental objects that do one of these things.

If you want to extract a specific spatiotemporal extent from a data file then
use :class:`Extracter`. If you want to split into a test and validation set
then use :class:`Extracter` to extract different temporal extents separately.
If you've already extracted data and written that to a file and then want to
sample that data for batches then use a :class:`Loader`, :class:`Sampler`, and
class:`SingleBatchQueue`. If you want to have training and validation batches
then load those separate data sets, wrap the data objects in Sampler objects
and provide these to :class:`BatchQueue`. If you want to have a BatchQueue
containing pairs of low / high res data, rather than coarsening high-res to get
low res then use :class:`DualBatchQueue` with :class:`DualSampler` objects.
"""

from .base import Container, DualContainer
from .batchers import (
    BatchHandlerCC,
    BatchHandlerDC,
    DualBatchQueue,
    SingleBatchQueue,
)
from .cachers import Cacher
from .collections import Collection, SamplerCollection, StatsCollection
from .derivers import Deriver
from .extracters import (
    BaseExtracterH5,
    BaseExtracterNC,
    DualExtracter,
    Extracter,
)
from .factories import (
    BatchHandler,
    DataHandlerH5,
    DataHandlerNC,
    DualBatchHandler,
    ExtracterH5,
    ExtracterNC,
)
from .loaders import Loader, LoaderH5, LoaderNC
from .samplers import (
    DataCentricSampler,
    DualSampler,
    Sampler,
)
from .wranglers import (
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
    DataHandlerNCforCC,
    DataHandlerNCforCCwithPowerLaw,
)
