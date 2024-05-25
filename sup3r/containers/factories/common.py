"""Objects common to factory output."""
from abc import ABCMeta


class FactoryMeta(ABCMeta, type):
    """Meta class to define __name__ attribute of factory generated classes."""

    def __new__(cls, name, bases, namespace, **kwargs):
        """Define __name__"""
        name = namespace.get("__name__", name)
        return super().__new__(cls, name, bases, namespace, **kwargs)
