"""Methods used by :class:`ForwardPass` and :class:`ForwardPassStrategy`"""
import logging

import sup3r.models

logger = logging.getLogger(__name__)


def get_model(model_class, kwargs):
    """Instantiate model after check on class name."""
    model_class = getattr(sup3r.models, model_class, None)
    if isinstance(kwargs, str):
        kwargs = {'model_dir': kwargs}

    if model_class is None:
        msg = (
            'Could not load requested model class "{}" from '
            'sup3r.models, Make sure you typed in the model class '
            'name correctly.'.format(model_class)
        )
        logger.error(msg)
        raise KeyError(msg)
    return model_class.load(**kwargs, verbose=True)
