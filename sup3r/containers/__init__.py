"""Top level containers. These are just things that have access to data.
Loaders, Handlers, Batchers, etc are subclasses of Containers. Rather than
having a single object that does everything - extract data, compute features,
sample the data for batching, split into train and val, etc, we have
fundamental objects that do one of these things.

If you want to extract a specific spatiotemporal extent from a data file then
use a class:`Wrangler`. If you want to split into a test and validation set
then use the Wrangler to extract different temporal extents separately. If
you've already extracted data and written that to a file and then want to
sample that data for batches then use a class:`Loader`, class:`Sampler`, and
class:`BatchQueue`. If you want to have training and validation batches then
load those separate data sets, wrap the data objects in Sampler objects and
provide these to class:`BatchQueueWithValidation`.
"""

from .base import Container, ContainerPair
