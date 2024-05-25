"""Methods used across container objects."""

import logging
from warnings import warn

logger = logging.getLogger(__name__)


def lowered(features):
    """Return a lower case version of the given str or list of strings. Used to
    standardize storage and lookup of features."""

    feats = (
        features.lower()
        if isinstance(features, str)
        else [f.lower() for f in features]
    )
    if features != feats:
        msg = (
            f'Received some upper case features: {features}. '
            f'Using {feats} instead.'
        )
        logger.warning(msg)
        warn(msg)
    return feats
