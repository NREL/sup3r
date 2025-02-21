sup3r.models.abstract.AbstractSingleModel
=========================================

.. currentmodule:: sup3r.models.abstract

.. autoclass:: AbstractSingleModel
   :members:
   :show-inheritance:
   :inherited-members:
   :special-members: __call__, __add__, __mul__

   
   
   .. rubric:: Methods

   .. autosummary::
   
      ~AbstractSingleModel.dict_to_tensorboard
      ~AbstractSingleModel.early_stop
      ~AbstractSingleModel.finish_epoch
      ~AbstractSingleModel.generate
      ~AbstractSingleModel.get_high_res_exo_input
      ~AbstractSingleModel.get_loss_fun
      ~AbstractSingleModel.get_optimizer_config
      ~AbstractSingleModel.get_optimizer_state
      ~AbstractSingleModel.get_single_grad
      ~AbstractSingleModel.init_optimizer
      ~AbstractSingleModel.load_network
      ~AbstractSingleModel.load_saved_params
      ~AbstractSingleModel.log_loss_details
      ~AbstractSingleModel.norm_input
      ~AbstractSingleModel.profile_to_tensorboard
      ~AbstractSingleModel.run_gradient_descent
      ~AbstractSingleModel.save
      ~AbstractSingleModel.set_norm_stats
      ~AbstractSingleModel.un_norm_output
      ~AbstractSingleModel.update_loss_details
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~AbstractSingleModel.generator
      ~AbstractSingleModel.generator_weights
      ~AbstractSingleModel.history
      ~AbstractSingleModel.means
      ~AbstractSingleModel.optimizer
      ~AbstractSingleModel.stdevs
      ~AbstractSingleModel.total_batches
   
   