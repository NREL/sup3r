"""Multi-threaded training session."""
import threading
from time import sleep


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

        self.train_thread.start()

        try:
            while True:
                sleep(0.01)
        except KeyboardInterrupt:
            self.train_thread.join()
            self.batch_handler.queue_thread.join()
            sleep(5.0)
            # self.batch_handler.stop()
