from abc import ABC
from dataclasses import dataclass
import logging

import numpy as np
import scipy

logger = logging.getLogger(__name__)


class BaseDistribution(ABC):
    def cdf(self, x):
        raise NotImplementedError("cdf was not implemented yet")

    def ppf(self, q):
        raise NotImplementedError("cdf was not implemented yet")

    def quantile_mapping(self, distribution):
        return lambda x: distribution.ppf(self.cdf(x))


@dataclass
class AnalyticalDistribution(BaseDistribution):
    """An well-known distribution with an analytical solution"""

    label: str
    params: tuple[float]

    def __post_init__(self):
        try:
            d = getattr(scipy.stats, self.label)
        except AttributeError:
            raise KeyError
        self._distribution = d(*self.params)

    def cdf(self, x):
        return self._distribution.cdf(x)

    def ppf(self, q):
        return self._distribution.ppf(q)


@dataclass
class EmpiricalDistribution(BaseDistribution):
    """An empirical distribution based on a given CDF"""

    quantile: tuple[float]
    cut_point: tuple[float]

    def __post_init__(self):
        # Check quantile and cut_point are:
        # - Both 1D
        # - Same size
        # - At least 2 points
        # - Both are ordered
        self._cdf = scipy.interpolate.interp1d(self.cut_point, self.quantile)
        self._ppf = scipy.interpolate.interp1d(self.quantile, self.cut_point)

    @staticmethod
    def from_fit(x, n_quantiles=21):
        """

        Example
        -------
        >>> MyDistribution = EmpiricalDistribution.from_fit((1,2,3,5))
        """
        q = np.linspace(0, 1, n_quantiles)
        cut_point = np.quantile(x, q)
        return EmpiricalDistribution(quantile=tuple(q), cut_point=tuple(cut_point))

    @staticmethod
    def from_quantiles(x):
        """

        Example
        -------
        >>> MyDistribution = EmpiricalDistribution.from_quantiles((1,2,3,5))
        """
        q = np.linspace(0, 1, len(x))
        return EmpiricalDistribution(quantile=tuple(q), cut_point=tuple(x))

    def cdf(self, x):
        return self._cdf(x)

    def ppf(self, q):
        return self._ppf(q)
