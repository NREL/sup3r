"""Multi-threaded training session."""
import threading


class TrainingSession:
    """Simple wrapper for multi-threaded training, with queued batching in the
    background."""

    def __init__(self, batch_handler, model, kwargs):
        self.model = model
        self.batch_handler = batch_handler
        self.kwargs = kwargs
        self.train_thread = threading.Thread(target=model.train,
                                             args=(batch_handler,),
                                             kwargs=kwargs)

        self.batch_handler.start()
        self.train_thread.start()

        self.train_thread.join()
        self.batch_handler.stop()
