"""Sup3r utilities"""

from enum import Enum


class ModuleName(str, Enum):
    """A collection of the module names available in sup3r.
    Each module name should match the name of the click command
    that will be used to invoke its respective cli. As of 5/26/2022,
    this means that all commands are lowercase with underscores
    replaced by dashes.
    """

    FORWARD_PASS = 'forward-pass'

    @classmethod
    def all_names(cls):
        """All module names.
        Returns
        -------
        set
            The set of all module name strings.
        """
        # pylint: disable=no-member
        return {v.value for v in cls.__members__.values()}
