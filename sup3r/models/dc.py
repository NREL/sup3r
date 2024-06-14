"""Sup3r data-centric model software"""

import json
import logging

import numpy as np

from sup3r.models.base import Sup3rGan
from sup3r.utilities.utilities import round_array

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
        list
            List of total losses for all sample bins
        """
        losses = []
        for batch in batch_handler.val_data:
            exo_data = self.get_high_res_exo_input(batch.high_res)
            gen = self._tf_generate(batch.low_res, exo_data)
            loss, _ = self.calc_loss(
                batch.high_res,
                gen,
                weight_gen_advers=weight_gen_advers,
                train_gen=True,
                train_disc=True,
            )
            losses.append(np.float32(loss))
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
        losses = []
        for batch in batch_handler.val_data:
            exo_data = self.get_high_res_exo_input(batch.high_res)
            gen = self._tf_generate(batch.low_res, exo_data)
            loss = self.calc_loss_gen_content(batch.high_res, gen)
            losses.append(np.float32(loss))
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

        if batch_handler.n_time_bins > 1:
            self.calc_bin_losses(
                total_losses,
                content_losses,
                batch_handler,
                loss_details,
                dim='time',
            )
        if batch_handler.n_space_bins > 1:
            self.calc_bin_losses(
                total_losses,
                content_losses,
                batch_handler,
                loss_details,
                dim='space',
            )

        loss_details['val_losses'] = json.dumps(
            round_array(total_losses)
        )
        return loss_details

    @staticmethod
    def calc_bin_losses(
        total_losses, content_losses, batch_handler, loss_details, dim
    ):
        """Calculate losses across spatial (temporal) samples and update
        corresponding weights. Spatial (temporal) weights are computed based on
        the temporal (spatial) averages of losses.

        Parameters
        ----------
        total_losses : array
            Array of total loss values across all validation sample bins
        content_losses : array
            Array of content loss values across all validation sample bins
        batch_handler : sup3r.preprocessing.BatchHandler
            BatchHandler object to iterate through
        loss_details : dict
            Namespace of the breakdown of loss components where each value is a
            running average at the current state in the epoch.
        dim : str
            Either 'time' or 'space'
        """
        msg = f'"dim" must be either "space" or "time", receieved {dim}'
        assert dim in ('time', 'space'), msg
        if dim == 'time':
            old_weights = batch_handler.temporal_weights.copy()
            axis = 0
        else:
            old_weights = batch_handler.spatial_weights.copy()
            axis = 1
        t_losses = (
            np.array(total_losses)
            .reshape((batch_handler.n_space_bins, batch_handler.n_time_bins))
            .mean(axis=axis)
        )
        t_c_losses = (
            np.array(content_losses)
            .reshape((batch_handler.n_space_bins, batch_handler.n_time_bins))
            .mean(axis=axis)
        )
        new_weights = t_losses / np.sum(t_losses)

        if dim == 'time':
            batch_handler.temporal_weights = new_weights
        else:
            batch_handler.spatial_weights = new_weights
        logger.debug(
            f'Previous {dim} bin weights: ' f'{round_array(old_weights)}'
        )
        logger.debug(f'{dim} losses (total): {round_array(t_losses)}')
        logger.debug(f'{dim} losses (content): ' f'{round_array(t_c_losses)}')
        logger.info(
            f'Updated {dim} bin weights: ' f'{round_array(new_weights)}'
        )
        loss_details[f'mean_{dim}_val_loss_gen'] = round(np.mean(t_losses), 3)
        loss_details[f'mean_{dim}_val_loss_gen_content'] = round(
            np.mean(t_c_losses), 3
        )
