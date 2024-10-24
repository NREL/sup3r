"""Sup3r data-centric model software"""

import logging

import numpy as np

from .base import Sup3rGan

np.set_printoptions(precision=3)

logger = logging.getLogger(__name__)


class Sup3rGanDC(Sup3rGan):
    """Data-centric model using loss across time bins to select training
    observations"""

    def calc_val_loss_gen(self, batch_handler, weight_gen_advers):
        """Calculate the validation total loss across the validation
        samples. e.g. If the sample domain has 100 steps and the
        validation set has 10 bins then this will get a list of losses across
        step 0 to 10, 10 to 20, etc.  Use this to determine performance
        within bins and to update how observations are selected from these
        bins.

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandlerDC
            BatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.

        Returns
        -------
        array
            Array of total losses for all sample bins, with shape
            (n_space_bins, n_time_bins)
        """
        losses = np.zeros(
            (batch_handler.n_space_bins, batch_handler.n_time_bins),
            dtype=np.float32,
        )
        for i, batch in enumerate(batch_handler.val_data):
            exo_data = self.get_high_res_exo_input(batch.high_res)
            loss, _ = self.calc_loss(
                hi_res_true=batch.high_res,
                hi_res_gen=self._tf_generate(batch.low_res, exo_data),
                weight_gen_advers=weight_gen_advers,
                train_gen=True,
                train_disc=True,
            )
            row = i // batch_handler.n_time_bins
            col = i % batch_handler.n_time_bins
            losses[row, col] = loss
        return losses

    def calc_val_loss_gen_content(self, batch_handler):
        """Calculate the validation content loss across the validation
        samples. e.g. If the sample domain has 100 steps and the
        validation set has 10 bins then this will get a list of losses across
        step 0 to 10, 10 to 20, etc.  Use this to determine performance
        within bins and to update how observations are selected from these
        bins.

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandlerDC
            BatchHandler object to iterate through

        Returns
        -------
        list
            List of content losses for all sample bins
        """
        losses = np.zeros(
            (batch_handler.n_space_bins, batch_handler.n_time_bins),
            dtype=np.float32,
        )
        for i, batch in enumerate(batch_handler.val_data):
            exo_data = self.get_high_res_exo_input(batch.high_res)
            loss = self.calc_loss_gen_content(
                hi_res_true=batch.high_res,
                hi_res_gen=self._tf_generate(batch.low_res, exo_data),
            )
            row = i // batch_handler.n_time_bins
            col = i % batch_handler.n_time_bins
            losses[row, col] = loss
        return losses

    def calc_val_loss(self, batch_handler, weight_gen_advers, loss_details):
        """Overloading the base calc_val_loss method. Method updates the
        temporal weights for the batch handler based on the losses across the
        time bins

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandler
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
        total_losses = self.calc_val_loss_gen(batch_handler, weight_gen_advers)
        content_losses = self.calc_val_loss_gen_content(batch_handler)

        t_weights = total_losses.mean(axis=0)
        t_weights /= t_weights.sum()

        s_weights = total_losses.mean(axis=1)
        s_weights /= s_weights.sum()

        logger.debug(
            f'Previous spatial weights: {batch_handler.spatial_weights}'
        )
        logger.debug(
            f'Previous temporal weights: {batch_handler.temporal_weights}'
        )
        batch_handler.update_weights(
            spatial_weights=s_weights, temporal_weights=t_weights
        )
        logger.debug(
            'New spatiotemporal weights (space, time):\n'
            f'{total_losses / total_losses.sum()}'
        )
        logger.debug(f'New spatial weights: {s_weights}')
        logger.debug(f'New temporal weights: {t_weights}')

        loss_details['mean_val_loss_gen'] = round(np.mean(total_losses), 3)
        loss_details['mean_val_loss_gen_content'] = round(
            np.mean(content_losses), 3
        )
        return loss_details
