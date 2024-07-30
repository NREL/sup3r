"""Utilities used by Loaders."""
import pandas as pd

from sup3r.preprocessing.names import Dimension


def lower_names(data):
    """Set all fields / coords / dims to lower case."""
    return data.rename(
        {
            f: f.lower()
            for f in [
                *list(data.data_vars),
                *list(data.dims),
                *list(data.coords),
            ]
            if f != f.lower()
        }
    )


def standardize_names(data, standard_names):
    """Standardize fields in the dataset using the `standard_names`
    dictionary."""
    data = lower_names(data)
    data = data.rename(
        {k: v for k, v in standard_names.items() if k in data}
    )
    return data


def standardize_values(data):
    """Standardize units and coordinate values. e.g. All temperatures in
    celsius, all longitudes between -180 and 180, etc.

    Note
    ----
    Currently (7/30/2024) only standarizes temperature units and coordinate
    values. Can add as needed.
    """
    for var in data.data_vars:
        attrs = data[var].attrs
        if 'units' in data[var].attrs and data[var].attrs['units'] == 'K':
            data[var] = (data[var].dims, data[var].values - 273.15)
            attrs['units'] = 'C'
        data[var].attrs = attrs

    data[Dimension.LONGITUDE] = (
        data[Dimension.LONGITUDE] + 180.0
    ) % 360.0 - 180.0
    if not data.time_independent:
        data[Dimension.TIME] = pd.to_datetime(data[Dimension.TIME])

    return data
