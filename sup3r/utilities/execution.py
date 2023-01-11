# -*- coding: utf-8 -*-
"""Execution methods for running some cli routines

@author: bbenton
"""

import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class DistributedProcess:
    """High-level class with commonly used functionality for processes
    distributed across multiple nodes"""

    def __init__(self, max_nodes=1, n_chunks=None, max_chunks=None,
                 incremental=False):
        """
        Parameters
        ----------
        max_nodes : int, optional
            Max number of nodes to distribute processes across
        n_chunks : int, optional
            Number of chunks to split all processes into. These process
            chunks will be distributed across nodes.
        max_chunks : int, optional
            Max number of chunks processes can be split into.
        incremental : bool
            Whether to skip previously run process chunks or to overwrite.
        """
        msg = ('For a distributed process either max_chunks or '
               'max_chunks + n_chunks must be specified. Received '
               f'max_chunks={max_chunks}, n_chunks={n_chunks}.')
        assert max_chunks is not None, msg
        self._node_chunks = None
        self._n_chunks = n_chunks
        self._max_nodes = max_nodes
        self._max_chunks = max_chunks
        self._out_files = None
        self._failed_chunks = False
        self.incremental = incremental

    def __len__(self):
        """Get total number of process chunks"""
        return self.chunks

    def node_finished(self, node_index):
        """Check if all out files for a given node have been saved

        Parameters
        ----------
        node_index : int
            Index of node to check for completed processes

        Returns
        -------
        bool
            Whether all processes for the given node have finished
        """
        return all(self.chunk_finished(i)
                   for i in self.node_chunks[node_index])

    # pylint: disable=E1136
    def chunk_finished(self, chunk_index):
        """Check if process for given chunk_index has already been run.

        Parameters
        ----------
        chunk_index : int
            Index of the process chunk to check for completion. Considered
            finished if there is already an output file and incremental is
            False.

        Returns
        -------
        bool
            Whether the process for the given chunk has finished
        """
        out_file = self.out_files[chunk_index]
        if os.path.exists(out_file) and self.incremental:
            logger.info('Not running chunk index {}, output file '
                        'exists: {}'.format(chunk_index, out_file))
            return True
        return False

    @property
    def all_finished(self):
        """Check if all out files have been saved"""
        return all(self.node_finished(i) for i in range(self.nodes))

    @property
    def out_files(self):
        """Get list of out files to write process output to"""
        return self._out_files

    @property
    def max_nodes(self):
        """Get uncapped max number of nodes to distribute processes across"""
        return self._max_nodes

    @property
    def chunks(self):
        """Get the number of process chunks for this distributed routine."""
        if self._n_chunks is None:
            return self._max_chunks
        else:
            return min(self._n_chunks, self._max_chunks)

    @property
    def nodes(self):
        """Get the max number of nodes to distribute chunks across, limited by
        the number of process chunks"""
        return len(self.node_chunks)

    @property
    def node_chunks(self):
        """Get the chunk indices for different nodes"""
        if self._node_chunks is None:
            n_chunks = min(self.max_nodes, self.chunks)
            self._node_chunks = np.array_split(np.arange(self.chunks),
                                               n_chunks)
        return self._node_chunks

    @property
    def failed_chunks(self):
        """Check whether any processes have failed."""
        return self._failed_chunks

    @failed_chunks.setter
    def failed_chunks(self, failed):
        """Set failed_chunks value. Should be set to True if there is a failed
        chunk"""
        self._failed_chunks = failed
