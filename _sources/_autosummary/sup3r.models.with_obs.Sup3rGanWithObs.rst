sup3r.models.with\_obs.Sup3rGanWithObs
======================================

.. currentmodule:: sup3r.models.with_obs

.. autoclass:: Sup3rGanWithObs
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
   
      ~Sup3rGanWithObs.calc_loss
      ~Sup3rGanWithObs.calc_loss_disc
      ~Sup3rGanWithObs.calc_loss_gen_content
      ~Sup3rGanWithObs.calc_val_loss
      ~Sup3rGanWithObs.check_batch_handler_attrs
      ~Sup3rGanWithObs.dict_to_tensorboard
      ~Sup3rGanWithObs.discriminate
      ~Sup3rGanWithObs.early_stop
      ~Sup3rGanWithObs.finish_epoch
      ~Sup3rGanWithObs.generate
      ~Sup3rGanWithObs.get_hr_exo_input
      ~Sup3rGanWithObs.get_loss_fun
      ~Sup3rGanWithObs.get_optimizer_config
      ~Sup3rGanWithObs.get_optimizer_state
      ~Sup3rGanWithObs.get_s_enhance_from_layers
      ~Sup3rGanWithObs.get_single_grad
      ~Sup3rGanWithObs.get_t_enhance_from_layers
      ~Sup3rGanWithObs.get_weight_update_fraction
      ~Sup3rGanWithObs.init_optimizer
      ~Sup3rGanWithObs.init_weights
      ~Sup3rGanWithObs.load
      ~Sup3rGanWithObs.load_network
      ~Sup3rGanWithObs.load_saved_params
      ~Sup3rGanWithObs.log_loss_details
      ~Sup3rGanWithObs.norm_input
      ~Sup3rGanWithObs.profile_to_tensorboard
      ~Sup3rGanWithObs.run_exo_layer
      ~Sup3rGanWithObs.run_gradient_descent
      ~Sup3rGanWithObs.save
      ~Sup3rGanWithObs.save_params
      ~Sup3rGanWithObs.seed
      ~Sup3rGanWithObs.set_model_params
      ~Sup3rGanWithObs.set_norm_stats
      ~Sup3rGanWithObs.train
      ~Sup3rGanWithObs.un_norm_output
      ~Sup3rGanWithObs.update_adversarial_weights
      ~Sup3rGanWithObs.update_loss_details
      ~Sup3rGanWithObs.update_optimizer
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Sup3rGanWithObs.discriminator
      ~Sup3rGanWithObs.discriminator_weights
      ~Sup3rGanWithObs.generator
      ~Sup3rGanWithObs.generator_weights
      ~Sup3rGanWithObs.history
      ~Sup3rGanWithObs.hr_exo_features
      ~Sup3rGanWithObs.hr_features
      ~Sup3rGanWithObs.hr_out_features
      ~Sup3rGanWithObs.input_dims
      ~Sup3rGanWithObs.input_resolution
      ~Sup3rGanWithObs.is_4d
      ~Sup3rGanWithObs.is_5d
      ~Sup3rGanWithObs.lr_features
      ~Sup3rGanWithObs.means
      ~Sup3rGanWithObs.meta
      ~Sup3rGanWithObs.model_params
      ~Sup3rGanWithObs.obs_features
      ~Sup3rGanWithObs.obs_training_inds
      ~Sup3rGanWithObs.optimizer
      ~Sup3rGanWithObs.optimizer_disc
      ~Sup3rGanWithObs.output_resolution
      ~Sup3rGanWithObs.s_enhance
      ~Sup3rGanWithObs.s_enhancements
      ~Sup3rGanWithObs.smoothed_features
      ~Sup3rGanWithObs.smoothing
      ~Sup3rGanWithObs.stdevs
      ~Sup3rGanWithObs.t_enhance
      ~Sup3rGanWithObs.t_enhancements
      ~Sup3rGanWithObs.total_batches
      ~Sup3rGanWithObs.version_record
      ~Sup3rGanWithObs.weights
   
   