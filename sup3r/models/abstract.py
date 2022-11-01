# -*- coding: utf-8 -*-
"""
Abstract class to define the required interface for Sup3r model subclasses
"""
import os
import json
from abc import ABC, abstractmethod
from phygnn import CustomNetwork


class AbstractSup3rGan(ABC):
    """
    Abstract class to define the required interface for Sup3r model subclasses

    Note that this only sets the required interfaces for a GAN that can be
    loaded from disk and used to predict synthetic outputs. The interface for
    models that can be trained will be set in another class.
    """

    @classmethod
    @abstractmethod
    def load(cls, model_dir, verbose=True):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        model_dir
            Directory to load GAN model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : BaseModel
            Returns a pretrained gan model that was previously saved to
            model_dir
        """

    @staticmethod
    def seed(s=0):
        """
        Set the random seed for reproducible results.

        Parameters
        ----------
        s : int
            Random seed
        """
        CustomNetwork.seed(s=s)

    @abstractmethod
    def generate(self, low_res):
        """Use the generator model to generate high res data from low res
        input. This is the public generate function.

        Parameters
        ----------
        low_res : np.ndarray
            Low-resolution input data, usually a 4D or 5D array of shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)

        Returns
        -------
        hi_res : ndarray
            Synthetically generated high-resolution data, usually a 4D or 5D
            array with shape:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """

    @property
    def s_enhance(self):
        """Factor by which model will enhance spatial resolution. Used in
        model training during high res coarsening"""
        return self.meta.get('s_enhance', None)

    @property
    def t_enhance(self):
        """Factor by which model will enhance temporal resolution. Used in
        model training during high res coarsening"""
        return self.meta.get('t_enhance', None)

    @property
    @abstractmethod
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

    @property
    @abstractmethod
    def training_features(self):
        """Get the list of input feature names that the generative model was
        trained on."""

    @property
    @abstractmethod
    def output_features(self):
        """Get the list of output feature names that the generative model
        outputs and that the discriminator predicts on."""

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        model_params = {'meta': self.meta}
        return model_params

    def save_params(self, out_dir):
        """
        Parameters
        ----------
        out_dir : str
            Directory to save linear model params. This directory will be
            created if it does not already exist.
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        fp_params = os.path.join(out_dir, 'model_params.json')
        with open(fp_params, 'w') as f:
            params = self.model_params
            json.dump(params, f, sort_keys=True, indent=2)
