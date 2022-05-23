# -*- coding: utf-8 -*-
"""Sup3r data-centric model software"""

import numpy as np
import time
import logging
import pandas as pd

from sup3r.models.modified_loss import Sup3rGanMMD
from sup3r.models.base import Sup3rGan

logger = logging.getLogger(__name__)


class Sup3rGanDC(Sup3rGan):
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
            List of total losses for all time bins
        """
        losses = []
        for obs in batch_handler.val_data:
            gen = self._tf_generate(obs.low_res)
            loss, _ = self.calc_loss(obs.high_res, gen,
                                     weight_gen_advers=weight_gen_advers,
                                     train_gen=True, train_disc=True)
            losses.append(loss)
        return losses

    def calc_time_bin_content_loss(self, batch_handler):
        """Calculate loss across time bins. e.g. Get the loss across time step
        0 to 100, 100 to 200, etc. Use this to determine performance within
        time bins and to update how observations are selected from these bins.
        Loss is calculated using content loss.

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through

        Returns
        -------
        list
            List of content losses for all time bins
        """
        losses = []
        for obs in batch_handler.val_data:
            gen = self._tf_generate(obs.low_res)
            loss = self.calc_loss_gen_content(obs.high_res, gen)
            losses.append(loss)
        return losses

    def update_temporal_weights(self, batch_handler, weight_gen_advers,
                                loss_details):
        """Update the temporal weights for the batch handler based on the
        losses across the time bins

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandler
            BatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        loss_details : dict
            Namespace of the breakdown of loss components where each value is a
            running average at the current state in the epoch.

        Returns
        -------
        dict
            Updated loss_details with mean validation loss calculated using
            the validation samples across the time bins
        """

        total_samples = batch_handler.batch_size * batch_handler.n_batches
        normalized_count = [round(float(s / total_samples), 3) for s
                            in batch_handler.training_sample_record]
        total_losses = self.calc_time_bin_loss(batch_handler,
                                               weight_gen_advers)
        content_losses = self.calc_time_bin_content_loss(batch_handler)
        new_temporal_weights = total_losses / np.sum(total_losses)
        previous_temporal_weights = batch_handler.temporal_weights
        batch_handler.temporal_weights = new_temporal_weights

        logger.debug('Sample count for temporal bins:'
                     f' {batch_handler.training_sample_record}')
        logger.debug(f'Previous normalized sampled count: {normalized_count}')
        logger.debug('Previous temporal bin weights: '
                     f'{[round(w, 3) for w in previous_temporal_weights]}')
        logger.debug('Temporal losses (total): '
                     f'{[round(float(tl), 3) for tl in total_losses]}')
        logger.debug('Temporal losses (content): '
                     f'{[round(float(cl), 3) for cl in content_losses]}')
        logger.info('Updated temporal bin weights: '
                    f'{[round(w, 3) for w in new_temporal_weights]}')

        loss_details['mean_val_loss'] = np.mean(total_losses)
        loss_details['mean_val_content_loss'] = np.mean(content_losses)
        return loss_details

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
              adaptive_update_bounds=(0.9, 0.99),
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

            logger.info('Epoch {} of {} '
                        'generator train loss: {:.2e}, '
                        'discriminator train loss: {:.2e} '
                        .format(epoch, epochs[-1],
                                loss_details['train_loss_gen'],
                                loss_details['train_loss_disc']
                                ))

            lr_g = self.get_optimizer_config(self.optimizer)['learning_rate']
            lr_d = self.get_optimizer_config(
                self.optimizer_disc)['learning_rate']

            extras = {'weight_gen_advers': weight_gen_advers,
                      'disc_loss_bound_0': disc_loss_bounds[0],
                      'disc_loss_bound_1': disc_loss_bounds[1],
                      'learning_rate_gen': lr_g,
                      'learning_rate_disc': lr_d}
            for i, w, in enumerate(batch_handler.temporal_weights):
                extras[f'temporal_weight_{i}'] = w

            loss_details = self.update_temporal_weights(
                batch_handler, weight_gen_advers, loss_details)

            weight_gen_advers = self.update_adversarial_weights(
                loss_details, adaptive_update_fraction, adaptive_update_bounds,
                weight_gen_advers, train_disc)

            stop = self.finish_epoch(epoch, epochs, t0, loss_details,
                                     checkpoint_int, out_dir,
                                     early_stop_on, early_stop_threshold,
                                     early_stop_n_epoch, extras=extras)

            if stop:
                break


class Sup3rGanDCwithMMD(Sup3rGanDC, Sup3rGanMMD):
    """Sup3rGan with MMD loss and Data centric observation selection"""
