# -*- coding: utf-8 -*-
"""Sup3r model software"""
import logging
import tensorflow as tf

from sup3r.models.base import Sup3rGan


logger = logging.getLogger(__name__)


class SolarCC(Sup3rGan):
    """Solar climate change model.

    Modifications to standard Sup3rGan:
        - Content loss is only on the n_days of the center 8 daylight hours of
          the daily true+synthetic high res samples
        - Discriminator only sees n_days of the center 8 daylight hours of the
          daily true high res sample.
        - Discriminator sees random n_days of 8-hour samples of the daily
          synthetic high res sample.
    """

    # starting hour is the hour that daylight starts at, daylight hours is the
    # number of daylight hours to sample, so for example if 8 and 8, the
    # daylight slice will be slice(8, 16)
    STARTING_HOUR = 8
    DAYLIGHT_HOURS = 8

    @tf.function
    def calc_loss(self, hi_res_true, hi_res_gen, weight_gen_advers=0.001,
                  train_gen=True, train_disc=False):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_gen : bool
            True if generator is being trained, then loss=loss_gen
        train_disc : bool
            True if disc is being trained, then loss=loss_disc

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the network being trained
            (either generator or one of the discriminators)
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        if hi_res_gen.shape != hi_res_true.shape:
            msg = ('The tensor shapes of the synthetic output {} and '
                   'true high res {} did not have matching shape! '
                   'Check the spatiotemporal enhancement multipliers in your '
                   'your model config and data handlers.'
                   .format(hi_res_gen.shape, hi_res_true.shape))
            logger.error(msg)
            raise RuntimeError(msg)

        msg = ('Special SolarCC model can only accept multi-day hourly '
               '(multiple of 24) high res data in the axis=3 position but '
               'received shape {}'.format(hi_res_true.shape))
        assert hi_res_true.shape[3] % 24 == 0

        msg = ('Special SolarCC model can only accept multi-day hourly '
               '(multiple of 24) high res synthetic data in the axis=3 '
               'position but received shape {}'.format(hi_res_gen.shape))
        assert hi_res_gen.shape[3] % 24 == 0

        t_len = hi_res_true.shape[3]
        n_days = int(t_len // 24)
        day_slices = [slice(self.STARTING_HOUR + x,
                            self.STARTING_HOUR + x + self.DAYLIGHT_HOURS)
                      for x in range(0, 24 * n_days, 24)]

        disc_out_true = []
        disc_out_gen = []
        loss_gen_content = 0.0
        for tslice in day_slices:
            disc_t = self._tf_discriminate(hi_res_true[:, :, :, tslice, :])
            gen_c = self.calc_loss_gen_content(hi_res_true[:, :, :, tslice, :],
                                               hi_res_gen[:, :, :, tslice, :])
            disc_out_true.append(disc_t)
            loss_gen_content += gen_c

        logits = [[1.0] * (t_len - self.DAYLIGHT_HOURS)]
        time_samples = tf.random.categorical(logits, len(day_slices))
        for i in range(len(day_slices)):
            t0 = time_samples[0, i]
            t1 = t0 + self.DAYLIGHT_HOURS
            disc_g = self._tf_discriminate(hi_res_gen[:, :, :, t0:t1, :])
            disc_out_gen.append(disc_g)

        disc_out_true = tf.concat([disc_out_true], axis=0)
        disc_out_gen = tf.concat([disc_out_gen], axis=0)
        loss_disc = self.calc_loss_disc(disc_out_true, disc_out_gen)

        loss_gen_content /= len(day_slices)
        loss_gen_advers = self.calc_loss_gen_advers(disc_out_gen)
        loss_gen = (loss_gen_content + weight_gen_advers * loss_gen_advers)

        loss = None
        if train_gen:
            loss = loss_gen
        elif train_disc:
            loss = loss_disc

        loss_details = {'loss_gen': loss_gen,
                        'loss_gen_content': loss_gen_content,
                        'loss_gen_advers': loss_gen_advers,
                        'loss_disc': loss_disc,
                        }

        return loss, loss_details
