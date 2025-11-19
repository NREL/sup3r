"""Utilities used by Loaders."""

from sup3r.preprocessing.names import Dimension
from sup3r.utilities.utilities import decode_str_times


def standardize_names(data, standard_names):
    """Standardize fields in the dataset using the `standard_names`
    dictionary."""
    data = data.rename({k: v for k, v in standard_names.items() if k in data})
    return data


def standardize_values(data):
    """Standardize units and coordinate values. e.g. All temperatures in
    celsius, all longitudes between -180 and 180, etc.

    data : xr.Dataset
        xarray dataset to be updated with standardized values.
    """
    for var in data.data_vars:
        attrs = data[var].attrs
        if 'units' in attrs and attrs['units'] == 'K':
            data.update({var: data[var] - 273.15})
            attrs['units'] = 'C'
        data[var].attrs.update(attrs)

    lons = (data[Dimension.LONGITUDE] + 180.0) % 360.0 - 180.0
    data[Dimension.LONGITUDE] = lons

    if Dimension.TIME in data.coords:
        data[Dimension.TIME] = decode_str_times(data[Dimension.TIME].values)

    return data
