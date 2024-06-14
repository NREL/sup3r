"""Container subclass with methods for sampling contained data.

TODO: With lazy loading / delayed calculations we could coarsen data prior to
sampling. This would allow us to use dual samplers for all cases, instead of
just for special paired datasets. This would be a nice unification.
"""

from .base import Sampler
from .cc import DualSamplerCC
from .dc import SamplerDC
from .dual import DualSampler
