# -*- coding: utf-8 -*-
"""Sup3r data-centric model software"""

import numpy as np
import logging
import json

from sup3r.models.base import Sup3rGan
from sup3r.utilities.utilities import round_array

logger = logging.getLogger(__name__)


class Sup3rGanDC(Sup3rGan):
    """Data-centric model using loss across time bins to select training
    observations"""

    def calc_temporal_val_loss_gen(self, batch_handler, weight_gen_advers):
        """Calculate the validation total loss across the temporal validation
        samples. e.g. If the temporal domain has 100 time steps and the
        validation set has 10 bins then this will get a list of losses across
        time step 0 to 10, 10 to 20, etc.  Use this to determine performance
        within time bins and to update how observations are selected from these
        bins.

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandlerDC
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
            losses.append(float(loss))
        return losses

    def calc_temporal_val_loss_gen_content(self, batch_handler):
        """Calculate the validation content loss across the temporal validation
        samples. e.g. If the temporal domain has 100 time steps and the
        validation set has 10 bins then this will get a list of losses across
        time step 0 to 10, 10 to 20, etc.  Use this to determine performance
        within time bins and to update how observations are selected from these
        bins.

        Parameters
        ----------
        batch_handler : sup3r.data_handling.preprocessing.BatchHandlerDC
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
            losses.append(float(loss))
        return losses

    def calc_val_loss(self, batch_handler, weight_gen_advers, loss_details):
        """Overloading the base calc_val_loss method. Method updates the
        temporal weights for the batch handler based on the losses across the
        time bins

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

        total_losses = self.calc_temporal_val_loss_gen(batch_handler,
                                                       weight_gen_advers)
        content_losses = self.calc_temporal_val_loss_gen_content(batch_handler)
        new_temporal_weights = total_losses / np.sum(total_losses)
        batch_handler.temporal_weights = new_temporal_weights

        logger.debug('Sample count for temporal bins:'
                     f' {batch_handler.training_sample_record}')
        logger.debug('Previous normalized sampled count: '
                     f'{round_array(batch_handler.normalized_sample_record)}')
        logger.debug(
            'Previous temporal bin weights: '
            f'{round_array(batch_handler.old_temporal_weights)}')
        logger.debug(f'Temporal losses (total): {round_array(total_losses)}')
        logger.debug('Temporal losses (content): '
                     f'{round_array(content_losses)}')
        logger.info('Updated temporal bin weights: '
                    f'{round_array(new_temporal_weights)}')

        loss_details['mean_temporal_val_loss_gen'] = round(np.mean(
            total_losses), 3)
        loss_details['mean_temporal_val_loss_gen_content'] = round(np.mean(
            content_losses), 3)
        loss_details['val_losses'] = json.dumps(round_array(total_losses))
        return loss_details
