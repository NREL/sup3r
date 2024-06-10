"""Abstract batch queue class used for conditional moment estimation."""

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import dask.array as da

from sup3r.models import Sup3rCondMom
from sup3r.preprocessing.batch_queues.base import SingleBatchQueue
from sup3r.typing import T_Array

logger = logging.getLogger(__name__)


@dataclass
class ConditionalBatch:
    """Conditional batch object, containing low_res, high_res, output, and mask
    data

    Parameters
    ----------
    low_res : T_Array
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    high_res : T_Array
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    output : T_Array
        Output predicted by the neural net. This can be different than the
        high_res when doing moment estimation. For ex: output may be
        (high_res)**2. We distinguish output from high_res since it may not be
        possible to recover high_res from output.
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    mask : T_Array
        Mask for the batch.
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    """

    low_res: T_Array
    high_res: T_Array
    output: T_Array
    mask: T_Array

    def __post_init__(self):
        self.shape = (self.low_res.shape, self.high_res.shape)

    def __len__(self):
        """Get the number of samples in this batch."""
        return len(self.low_res)


class ConditionalBatchQueue(SingleBatchQueue):
    """BatchQueue class for conditional moment estimation."""

    BATCH_CLASS = ConditionalBatch

    def __init__(
        self,
        *args,
        time_enhance_mode: Optional[str] = 'constant',
        model_mom1: Optional[Sup3rCondMom] = None,
        s_padding: Optional[int] = None,
        t_padding: Optional[int] = None,
        end_t_padding: Optional[bool] = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        *args : list
            Positional arguments for parent class
        time_enhance_mode : str
            [constant, linear]
            Method to enhance temporally when constructing subfilter.  At every
            temporal location, a low-res temporal data is substracted from the
            high-res temporal data predicted.  constant will assume that the
            low-res temporal data is constant between landmarks.  linear will
            linearly interpolate between landmarks to generate the low-res data
            to remove from the high-res.
        model_mom1 : Sup3rCondMom | None
            model that predicts the first conditional moments.  Useful to
            prepare data for learning second conditional moment.
        s_padding : int | None
            Width of spatial padding to predict only middle part. If None, no
            padding is used
        t_padding : int | None
            Width of temporal padding to predict only middle part. If None, no
            padding is used
        end_t_padding : bool | False
            Zero pad the end of temporal space.  Ensures that loss is
            calculated only if snapshot is surrounded by temporal landmarks.
            False by default
        **kwargs : dict
            Keyword arguments for parent class
        """
        self.low_res = None
        self.high_res = None
        self.output = None
        self.s_padding = s_padding
        self.t_padding = t_padding
        self.end_t_padding = end_t_padding
        self.time_enhance_mode = time_enhance_mode
        self.model_mom1 = model_mom1
        super().__init__(*args, **kwargs)

    def make_mask(self, high_res):
        """Make mask for output. This is used to ensure consistency when
        training conditional moments.

        Note
        ----
        Consider the case of learning E(HR|LR) where HR is the high_res and LR
        is the low_res. In theory, the conditional moment estimation works if
        the full LR is passed as input and predicts the full HR. In practice,
        only the LR data that overlaps and surrounds the HR data is useful, ie
        E(HR|LR) = E(HR|LR_nei) where LR_nei is the LR data that surrounds the
        HR data. Physically, this is equivalent to saying that data far away
        from a region of interest does not matter.  This allows learning the
        conditional moments on spatial and temporal chunks only if one
        restricts the high_res output as being overlapped and surrounded by the
        input low_res.  The role of the mask is to ensure that the input
        low_res always surrounds the output high_res.

        Parameters
        ----------
        high_res : T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        mask: T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """
        mask = da.zeros(high_res)
        s_min = self.s_padding if self.s_padding is not None else 0
        t_min = self.t_padding if self.t_padding is not None else 0
        s_max = -self.s_padding if s_min > 0 else None
        t_max = -self.t_padding if t_min > 0 else None
        if self.end_t_padding and self.t_enhance > 1:
            if t_max is None:
                t_max = -(self.t_enhance - 1)
            else:
                t_max = -(self.t_enhance - 1) - self.t_padding

        if len(high_res.shape) == 4:
            mask[:, s_min:s_max, s_min:s_max, :] = 1.0
        elif len(high_res.shape) == 5:
            mask[:, s_min:s_max, s_min:s_max, t_min:t_max, :] = 1.0

        return mask

    @abstractmethod
    def make_output(self, samples):
        """Make custom batch output. This depends on the moment type being
        estimated. e.g. This could be the 1st moment, which is just high_res
        or the 2nd moment, which is (high_res - 1st moment) ** 2

        Parameters
        ----------
        samples : Tuple[T_Array, T_Array]
            Tuple of low_res, high_res. Each array is:
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        output: T_Array
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """

    def batch_next(self, samples):
        """Returns normalized collection of samples / observations along with
        mask and target output for conditional moment estimation. Performs
        coarsening on high-res data if :class:`Collection` consists of
        :class:`Sampler` objects and not :class:`DualSampler` objects

        Returns
        -------
        Batch
            Batch object with `low_res`, `high_res`, `mask`, and `output`
            attributes
        """
        lr, hr = self.transform(samples, **self.transform_kwargs)
        lr, hr = self.normalize(lr, hr)
        mask = self.make_mask(high_res=hr)
        output = self.make_output(samples=(lr, hr))
        return self.BATCH_CLASS(
            low_res=lr, high_res=hr, output=output, mask=mask
        )
