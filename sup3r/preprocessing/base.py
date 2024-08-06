"""Base classes - fundamental dataset objects and the base :class:`Container`
object, which just contains dataset objects. All objects that interact with
data are containers. e.g. loaders, rasterizers, data handlers, samplers, batch
queues, batch handlers.
"""

import logging
import pprint
from abc import ABCMeta
from collections import namedtuple
from typing import Optional, Tuple, Union
from warnings import warn

import numpy as np
import xarray as xr

import sup3r.preprocessing.accessor  # noqa: F401 # pylint: disable=W0611
from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.utilities import composite_info

logger = logging.getLogger(__name__)


def _get_class_info(namespace):
    sig_objs = namespace.get('_signature_objs', None)
    skips = namespace.get('_skip_params', None)
    _sig = _doc = None
    if sig_objs:
        _sig, _doc = composite_info(sig_objs, skip_params=skips)
    return _sig, _doc


class Sup3rMeta(ABCMeta, type):
    """Meta class to define ``__name__``, ``__signature__``, and
    ``__subclasscheck__`` of composite and derived classes. This allows us to
    still resolve a signature for classes which pass through parent args /
    kwargs as ``*args`` / ``**kwargs`` or those built through factory
    composition, for example."""

    def __new__(mcs, name, bases, namespace, **kwargs):  # noqa: N804
        """Define __name__ and __signature__"""
        _sig, _doc = _get_class_info(namespace)
        name = namespace.get('__name__', name)
        if _sig:
            namespace['__signature__'] = _sig
        if '__init__' in namespace and _sig:
            namespace['__init__'].__signature__ = _sig
        if '__init__' in namespace and _doc:
            namespace['__init__'].__doc__ = _doc
        return super().__new__(mcs, name, bases, namespace, **kwargs)

    def __subclasscheck__(cls, subclass):
        """Check if factory built class shares base classes."""
        if super().__subclasscheck__(subclass):
            return True
        if hasattr(subclass, '_signature_objs'):
            return {obj.__name__ for obj in cls._signature_objs} == {
                obj.__name__ for obj in subclass._signature_objs
            }
        return False

    def __repr__(cls):
        return f"<class '{cls.__module__}.{cls.__name__}'>"


class Sup3rDataset:
    """Interface for interacting with one or two ``xr.Dataset`` instances.
    This is either a simple passthrough for a ``xr.Dataset`` instance or a
    wrapper around two of them so they work well with Dual objects like
    ``DualSampler``, ``DualRasterizer``, ``DualBatchHandler``, etc...)

    Examples
    --------
    >>> hr = xr.Dataset(...)
    >>> lr = xr.Dataset(...)
    >>> ds = Sup3rDataset(low_res=lr, high_res=hr)
    >>> # access high_res or low_res:
    >>> ds.high_res; ds.low_res

    >>> daily = xr.Dataset(...)
    >>> hourly = xr.Dataset(...)
    >>> ds = Sup3rDataset(daily=daily, hourly=hourly)
    >>> # access hourly or daily:
    >>> ds.hourly; ds.daily

    Note
    ----
    (1) This may seem similar to
    :class:`~sup3r.preprocessing.collections.base.Collection`, which also can
    contain multiple data members, but members of
    :class:`~sup3r.preprocessing.collections.base.Collection` objects are
    completely independent while here there are at most two members which are
    related as low / high res versions of the same underlying data.

    (2) Here we make an important choice to use high_res members to compute
    means / stds. It would be reasonable to instead use the average of high_res
    and low_res means / stds for aggregate stats but we want to preserve the
    relationship between coarsened variables after normalization (e.g.
    temperature_2m, temperature_max_2m, temperature_min_2m). This means all
    these variables should have the same means and stds, which ultimately come
    from the high_res non coarsened variable.
    """

    def __init__(
        self,
        data: Optional[
            Union[Tuple[xr.Dataset, ...], Tuple[Sup3rX, ...]]
        ] = None,
        **dsets: Union[xr.Dataset, Sup3rX],
    ):
        """
        Parameters
        ----------
        data : Tuple[xr.Dataset | Sup3rX | Sup3rDataset]
            ``Sup3rDataset`` will accomodate various types of data inputs,
            which will ultimately be wrapped as a namedtuple of
            :class:`~sup3r.preprocessing.Sup3rX` objects, stored in the
            self._ds attribute. The preferred way to pass data here is through
            dsets, as a dictionary with names. If data is given as a tuple of
            :class:`~sup3r.preprocessing.Sup3rX` objects then great, no prep
            needed. If given as a tuple of ``xr.Dataset`` objects then each
            will be cast to ``Sup3rX`` objects. If given as tuple of
            Sup3rDataset objects then we make sure they contain only a single
            data member and use those to initialize a new ``Sup3rDataset``.

            If the tuple here is a singleton the namedtuple will use the name
            "high_res" for the single dataset. If the tuple is a doublet then
            the first tuple member will be called "low_res" and the second
            will be called "high_res".

        dsets : dict[str, Union[xr.Dataset, Sup3rX]]
            The preferred way to initialize a ``Sup3rDataset`` object, as a
            dictionary with keys used to name a namedtuple of ``Sup3rX``
            objects. If dsets contains xr.Dataset objects these will be cast
            to ``Sup3rX`` objects first.

        """
        if data is not None:
            data = data if isinstance(data, tuple) else (data,)
            if all(isinstance(d, type(self)) for d in data):
                msg = (
                    'Sup3rDataset received a tuple of Sup3rDataset objects'
                    ', each with two data members. If you insist on '
                    'initializing a Sup3rDataset with a tuple of the same, '
                    'then they have to be singletons.'
                )
                assert all(len(d) == 1 for d in data), msg
                msg = (
                    'Sup3rDataset received a tuple of Sup3rDataset '
                    'objects. You got away with it this time because they '
                    'each contain a single data member, but be careful'
                )
                logger.warning(msg)
                warn(msg)

            if len(data) == 1:
                msg = (
                    f'{self.__class__.__name__} received a single data member '
                    'without an explicit name. Interpreting this as '
                    '(high_res,). To be explicit provide keyword arguments '
                    'like Sup3rDataset(high_res=data[0])'
                )
                logger.warning(msg)
                warn(msg)
                dsets = {'high_res': data[0]}
            elif len(data) == 2:
                msg = (
                    f'{self.__class__.__name__} received a data tuple. '
                    'Interpreting this as (low_res, high_res). To be explicit '
                    'provide keyword arguments like '
                    'Sup3rDataset(low_res=data[0], high_res=data[1])'
                )
                logger.warning(msg)
                warn(msg)
                dsets = {'low_res': data[0], 'high_res': data[1]}
            else:
                msg = (
                    f'{self.__class__.__name__} received tuple of length '
                    f'{len(data)}. Can only handle 1 / 2 - tuples.'
                )
                logger.error(msg)
                raise ValueError(msg)

        dsets = {
            k: Sup3rX(v)
            if isinstance(v, xr.Dataset)
            else v._ds[0]
            if isinstance(v, type(self))
            else v
            for k, v in dsets.items()
        }
        self._ds = namedtuple('Dataset', list(dsets))(**dsets)

    def __iter__(self):
        yield from self._ds

    @property
    def dtype(self):
        """Get datatype of first member. Assumed to be constant for all
        members."""
        return self._ds[0].dtype

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, attr):
        """Get attribute through accessor if available. Otherwise use standard
        xarray interface."""
        if hasattr(self._ds, attr):
            return getattr(self._ds, attr)
        out = [self._getattr(ds, attr) for ds in self._ds]
        if len(self._ds) == 1:
            out = out[0]
        return out

    def _getattr(self, dset, attr):
        """Get attribute from single data member."""
        return (
            getattr(dset.sx, attr)
            if hasattr(dset.sx, attr)
            else getattr(dset, attr)
        )

    def _getitem(self, dset, item):
        """Get item from single data member."""
        return dset.sx[item] if hasattr(dset, 'sx') else dset[item]

    def get_dual_item(self, keys):
        """Method for getting items from self._ds when it consists of two
        datasets. If keys is a `List[Tuple]` or `List[List]` this is
        interpreted as a request for `self._ds[i][keys[i]] for i in
        range(len(keys)).` Otherwise we will get keys from each member of
        self.dset.

        Note
        ----
        This casts back to `type(self)` before final return if result of get
        item from each member of `self._ds` is a tuple of `Sup3rX` instances
        """
        if isinstance(keys, (tuple, list)) and all(
            isinstance(k, (tuple, list)) for k in keys
        ):
            out = tuple(
                self._getitem(d, key) for d, key in zip(self._ds, keys)
            )
        else:
            out = tuple(self._getitem(d, keys) for d in self._ds)
        return (
            type(self)(**dict(zip(self._ds._fields, out)))
            if all(isinstance(o, Sup3rX) for o in out)
            else out
        )

    def rewrap(self, data):
        """Rewrap data as Sup3rDataset after calling parent method."""
        if isinstance(data, type(self)):
            return data
        return (
            type(self)(low_res=data[0], high_res=data[1])
            if len(data) > 1
            else type(self)(high_res=data[0])
        )

    def sample(self, idx):
        """Get samples from self._ds members. idx should be either a tuple of
        slices for the dimensions (south_north, west_east, time) and a list of
        feature names or a 2-tuple of the same, for dual datasets."""
        if len(self._ds) == 2:
            return tuple(d.sample(idx[i]) for i, d in enumerate(self))
        return self._ds[-1].sample(idx)

    def isel(self, *args, **kwargs):
        """Return new Sup3rDataset with isel applied to each member."""
        return self.rewrap(tuple(d.isel(*args, **kwargs) for d in self))

    def __getitem__(self, keys):
        """If keys is an int this is interpreted as a request for that member
        of self._ds. If self._ds consists of two members we call
        :py:meth:`~sup3r.preprocesing.Sup3rDataset.get_dual_item`. Otherwise we
        get the item from the single member of self._ds."""
        if isinstance(keys, int):
            return self._ds[keys]
        if len(self._ds) == 1:
            return self._ds[-1][keys]
        return self.get_dual_item(keys)

    @property
    def shape(self):
        """We use the shape of the largest data member. These are assumed to be
        ordered as (low-res, high-res) if there are two members."""
        return self._ds[-1].shape

    @property
    def features(self):
        """The features are determined by the set of features from all data
        members."""
        feats = [
            f for f in self._ds[0].features if f not in self._ds[-1].features
        ]
        feats += self._ds[-1].features
        return feats

    @property
    def size(self):
        """Return number of elements in the largest data member."""
        return np.prod(self.shape)

    def __contains__(self, vals):
        """Check for vals in all of the dset members."""
        return any(d.sx.__contains__(vals) for d in self._ds)

    def __setitem__(self, keys, data):
        """Set dset member values. Check if values is a tuple / list and if
        so interpret this as sending a tuple / list element to each dset
        member. e.g. `vals[0] -> dsets[0]`, `vals[1] -> dsets[1]`, etc"""
        if len(self._ds) == 1:
            self._ds[-1].__setitem__(keys, data)
        else:
            for i, self_i in enumerate(self):
                dat = data[i] if isinstance(data, (tuple, list)) else data
                self_i.__setitem__(keys, dat)

    def mean(self, **kwargs):
        """Use the high_res members to compute the means. These are used for
        normalization during training."""
        kwargs['skipna'] = kwargs.get('skipna', True)
        return self._ds[-1].mean(**kwargs)

    def std(self, **kwargs):
        """Use the high_res members to compute the stds. These are used for
        normalization during training."""
        kwargs['skipna'] = kwargs.get('skipna', True)
        return self._ds[-1].std(**kwargs)

    def normalize(self, means, stds):
        """Normalize dataset using the given mean and stds. These are provided
        as dictionaries."""
        _ = [d.normalize(means=means, stds=stds) for d in self._ds]

    def compute(self, **kwargs):
        """Load data into memory for each data member."""
        _ = [d.compute(**kwargs) for d in self._ds]

    @property
    def loaded(self):
        """Check if all data members have been loaded into memory."""
        return all(d.loaded for d in self._ds)


class Container(metaclass=Sup3rMeta):
    """Basic fundamental object used to build preprocessing objects. Contains
    an xarray-like Dataset (:class:`~.accessor.Sup3rX`), wrapped tuple of
    `Sup3rX` objects (:class:`.Sup3rDataset`), or a tuple of such objects.
    """

    __slots__ = ['_data']

    def __init__(
        self,
        data: Union[
            Sup3rX, Sup3rDataset, Tuple[Sup3rX, ...], Tuple[Sup3rDataset, ...]
        ] = None,
    ):
        """
        Parameters
        ----------
        data: Union[Sup3rX, Sup3rDataset, Tuple[Sup3rX, ...],
                    Tuple[Sup3rDataset, ...]
            Can be an ``xr.Dataset``, a :class:`~.accessor.Sup3rX` object, a
            :class:`.Sup3rDataset` object, or a tuple of such objects.

            Note
            ----
            ``.data`` will return a :class:`~.Sup3rDataset` object or tuple of
            such. This is a tuple when the `.data` attribute belongs to a
            :class:`~.collections.base.Collection` object like
            :class:`~.batch_handlers.factory.BatchHandler`. Otherwise this is
            :class:`~.Sup3rDataset` object, which is either a wrapped 2-tuple
            or 1-tuple (e.g. ``len(data) == 2`` or ``len(data) == 1)``. This is
            a 2-tuple when ``.data`` belongs to a dual container object like
            :class:`~.samplers.DualSampler` and a 1-tuple otherwise.
        """
        self.data = data

    @property
    def data(self):
        """Return underlying data.

        See Also
        --------
        :py:meth:`.wrap`
        """
        return self._data

    @data.setter
    def data(self, data):
        """Set data value. Wrap given value depending on type.

        See Also
        --------
        :py:meth:`.wrap`"""
        self._data = self.wrap(data)

    @staticmethod
    def wrap(data):
        """Return a :class:`~.Sup3rDataset` object or tuple of such. This is a
        tuple when the `.data` attribute belongs to a
        :class:`~sup3r.preprocessing.collections.Collection` object like
        :class:`~sup3r.preprocessing.batch_handlers.BatchHandler`. Otherwise
        this is is :class:`~.Sup3rDataset` objects, which is either a wrapped
        2-tuple or 1-tuple (e.g. `len(data) == 2` or `len(data) == 1`)
        depending on whether this container is used for a dual dataset or not.
        """
        if isinstance(data, Sup3rDataset):
            return data
        if isinstance(data, tuple) and all(
            isinstance(d, Sup3rDataset) for d in data
        ):
            return data
        return (
            Sup3rDataset(low_res=data[0], high_res=data[1])
            if isinstance(data, tuple) and len(data) == 2
            else Sup3rDataset(high_res=data)
            if data is not None and not isinstance(data, Sup3rDataset)
            else data
        )

    def post_init_log(self, args_dict=None):
        """Log additional arguments after initialization."""
        if args_dict is not None:
            logger.info(
                f'Finished initializing {self.__class__.__name__} with:\n'
                f'{pprint.pformat(args_dict, indent=2)}'
            )

    @property
    def shape(self):
        """Get shape of underlying data."""
        return self.data.shape

    def __contains__(self, vals):
        return vals in self.data

    def __getitem__(self, keys):
        """Get item from underlying data. ``.data`` is a ``Sup3rX`` or
        ``Sup3rDataset`` object, so this uses those ``__getitem__`` methods.

        See Also
        --------
        :py:meth:`.accessor.Sup3rX.__getitem__`,
        :py:meth:`.Sup3rDataset.__getitem__`
        """
        return self.data[keys]

    def __setitem__(self, keys, data):
        """Set item in underlying data."""
        self.data.__setitem__(keys, data)

    def __getattr__(self, attr):
        """Check if attribute is available from ``.data``"""
        try:
            data = self.__getattribute__('_data')
            return getattr(data, attr)
        except Exception as e:
            msg = f'{self.__class__.__name__} object has no attribute "{attr}"'
            raise AttributeError(msg) from e
