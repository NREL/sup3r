# -*- coding: utf-8 -*-
"""Sup3r data-centric model software"""

import numpy as np
import time
import logging
import pandas as pd

from sup3r.models.modified_loss import Sup3rGanMMD

logger = logging.getLogger(__name__)


class Sup3rGanDC(Sup3rGanMMD):
    """Data-centric model using loss across time bins to select training
    observations"""

    def calc_time_bin_loss(self, batch_handler, weight_gen_advers):
        """Calculate loss across time bins. e.g. Get the loss across time step
        0 to 100, 100 to 200, etc. Use this to determine performance within
        time bins and to update how observations are selected from these bins.
        Loss is calculated using total loss.

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.

        Returns
        -------
        list
            List of losses for all time bins
        """
        losses = []
        for obs in batch_handler.val_data:
            gen = self._tf_generate(obs.low_res)
            loss, _ = self.calc_loss(obs.high_res, gen,
                                     weight_gen_advers=weight_gen_advers,
                                     train_gen=True, train_disc=True)
            losses.append(loss)
        return losses

    def train(self, batch_handler, n_epoch,
              weight_gen_advers=0.001,
              train_gen=True,
              train_disc=True,
              disc_loss_bounds=(0.45, 0.6),
              checkpoint_int=None,
              out_dir='./gan_{epoch}',
              early_stop_on=None,
              early_stop_threshold=0.005,
              early_stop_n_epoch=5,
              adaptive_update_bounds=(0.5, 0.95),
              adaptive_update_fraction=0.05):
        """Train the GAN model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        n_epoch : int
            Number of epochs to train on
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
        disc_loss_bounds : tuple
            Lower and upper bounds for the discriminator loss outside of which
            the discriminator will not train unless train_disc=True and
            train_gen=False.
        checkpoint_int : int | None
            Epoch interval at which to save checkpoint models.
        out_dir : str
            Directory to save checkpoint GAN models. Should have {epoch} in
            the directory name. This directory will be created if it does not
            already exist.
        early_stop_on : str | None
            If not None, this should be a column in the training history to
            evaluate for early stopping (e.g. validation_loss_gen,
            validation_loss_disc). If this value in this history decreases by
            an absolute fractional relative difference of less than 0.01 for
            more than 5 epochs in a row, the training will stop early.
        early_stop_threshold : float
            The absolute relative fractional difference in validation loss
            between subsequent epochs below which an early termination is
            warranted. E.g. if val losses were 0.1 and 0.0998 the relative
            diff would be calculated as 0.0002 / 0.1 = 0.002 which would be
            less than the default thresold of 0.01 and would satisfy the
            condition for early termination.
        early_stop_n_epoch : int
            The number of consecutive epochs that satisfy the threshold that
            warrants an early stop.
        adaptive_update_bounds : tuple
            Tuple specifying allowed range for loss_details[comparison_key]. If
            history[comparison_key] < threshold_range[0] then the weight will
            be increased by (1 + update_frac). If history[comparison_key] >
            threshold_range[1] then the weight will be decreased by 1 / (1 +
            update_frac).
        adaptive_update_fraction : float
            Amount by which to increase or decrease adversarial weights for
            adaptive updates
        """

        self.set_norm_stats(batch_handler)
        self.set_feature_names(batch_handler)

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(
                columns=['elapsed_time'])
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        t0 = time.time()
        logger.info('Training model with adversarial weight: {} '
                    'for {} epochs starting at epoch {}'
                    .format(weight_gen_advers, n_epoch, epochs[0]))

        for epoch in epochs:
            loss_details = self.train_epoch(batch_handler, weight_gen_advers,
                                            train_gen, train_disc,
                                            disc_loss_bounds)

            temporal_losses = self.calc_time_bin_loss(batch_handler,
                                                      weight_gen_advers)
            logger.info(
                f'Temporal losses: {[round(tl, 3) for tl in temporal_losses]}')

            new_temporal_weights = temporal_losses / np.sum(temporal_losses)
            batch_handler.update_temporal_weights(new_temporal_weights)

            loss_details['mean_val_loss'] = np.mean(temporal_losses)

            logger.info('Epoch {} of {} '
                        'generator train loss: {:.2e} '
                        'discriminator train loss: {:.2e} '
                        'mean gen val loss: {:.2e}'
                        .format(epoch, epochs[-1],
                                loss_details['train_loss_gen'],
                                loss_details['train_loss_disc'],
                                loss_details['mean_val_loss'],
                                ))

            lr_g = self.get_optimizer_config(self.optimizer)['learning_rate']
            lr_d = self.get_optimizer_config(
                self.optimizer_disc)['learning_rate']

            extras = {'weight_gen_advers': weight_gen_advers,
                      'disc_loss_bound_0': disc_loss_bounds[0],
                      'disc_loss_bound_1': disc_loss_bounds[1],
                      'learning_rate_gen': lr_g,
                      'learning_rate_disc': lr_d}

            stop = self.finish_epoch(epoch, epochs, t0, loss_details,
                                     checkpoint_int, out_dir,
                                     early_stop_on, early_stop_threshold,
                                     early_stop_n_epoch, extras=extras)

            weight_gen_advers = self.update_adversarial_weights(
                self.history, adaptive_update_fraction, adaptive_update_bounds,
                weight_gen_advers, train_disc)

            if stop:
                break
