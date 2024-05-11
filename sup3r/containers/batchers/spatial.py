"""Batch queue objects for training spatial only models."""


from sup3r.containers.batchers.base import BatchQueue


class SpatialBatchQueue(BatchQueue):
    """Sup3r spatial batch handling class"""

    def get_next(self):
        """Remove time dimension since this is a batcher for a spatial only
        model."""
        samples = self.queue.dequeue()
        batch = self.batch_next(samples[..., 0, :])
        return batch
