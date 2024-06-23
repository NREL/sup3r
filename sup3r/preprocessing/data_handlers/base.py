"""Base container classes - object that contains data. All objects that
interact with data are containers. e.g. loaders, extracters, data handlers,
samplers, batch queues, batch handlers.
"""

import logging

logger = logging.getLogger(__name__)


class SingleExoDataStep(dict):
    """Special dictionary class for exogenous_data step"""

    def __init__(self, feature, combine_type, model, data):
        """exogenous_data step dictionary for a given model step

        Parameters
        ----------
        feature : str
            Name of feature corresponding to `data`.
        combine_type : str
            Specifies how the exogenous_data should be used for this step. e.g.
            "input", "layer", "output". For example, if tis equals "input" the
            `data` will be used as input to the forward pass for the model step
            given by `model`
        model : int
            Specifies the model index which will use the `data`. For example,
            if `model` == 1 then the `data` will be used according to
            `combine_type` in the 2nd model step in a MultiStepGan.
        data : tf.Tensor | np.ndarray
            The data to be used for the given model step.
        """
        step = {'model': model, 'combine_type': combine_type, 'data': data}
        for k, v in step.items():
            self.__setitem__(k, v)
        self.feature = feature

    @property
    def shape(self):
        """Shape of data array for this model step."""
        return self['data'].shape


class ExoData(dict):
    """Special dictionary class for multiple exogenous_data steps

    TODO: Can we simplify this by relying more on xr.Dataset meta data instead
    of storing enhancement factors for each step? Seems like we could take the
    highest res data and coarsen based on s/t enhance, also.
    """

    def __init__(self, steps):
        """Combine multiple SingleExoDataStep objects

        Parameters
        ----------
        steps : dict
            Dictionary with feature keys each with entries describing whether
            features should be combined at input, a mid network layer, or with
            output. e.g.
            {'topography': {'steps': [
                {'combine_type': 'input', 'model': 0, 'data': ...,
                 'resolution': ...},
                {'combine_type': 'layer', 'model': 0, 'data': ...,
                 'resolution': ...}]}}
            Each array in in 'data' key has 3D or 4D shape:
            (spatial_1, spatial_2, 1)
            (spatial_1, spatial_2, n_temporal, 1)
        """
        if isinstance(steps, dict):
            for k, v in steps.items():
                self.__setitem__(k, v)
        else:
            msg = 'ExoData must be initialized with a dictionary of features.'
            logger.error(msg)
            raise ValueError(msg)

    def get_model_step_exo(self, model_step):
        """Get the exogenous data for the given model_step from the full list
        of steps

        Parameters
        ----------
        model_step : int
            Index of the model to get exogenous data for.

        Returns
        -------
        model_step_exo : dict
            Dictionary of features each with list of steps which match the
            given model_step
        """
        model_step_exo = {}
        for feature, entry in self.items():
            steps = [
                step for step in entry['steps'] if step['model'] == model_step
            ]
            if steps:
                model_step_exo[feature] = {'steps': steps}
        return ExoData(model_step_exo)

    def split(self, split_steps):
        """Split `self` into multiple dicts based on split_steps. The splits
        are done such that the steps in the ith entry of the returned list
        all have a `model number < split_steps[i].`

        TODO: lots of nested loops here. simplify the logic.

        Parameters
        ----------
        split_steps : list
            Step index list to use for splitting. To split this into exo data
            for spatial models and temporal models split_steps should be
            [len(spatial_models)]. If this is for a TemporalThenSpatial model
            split_steps should be [len(temporal_models)]. If this is for a
            multi step model composed of more than two models (e.g.
            SolarMultiStepGan) split_steps should be
            [len(spatial_solar_models), len(spatial_solar_models) +
            len(spatial_wind_models)]

        Returns
        -------
        split_list : List[ExoData]
            List of `ExoData` objects coming from the split of `self`,
            according to `split_steps`
        """
        split_dict = {i: {} for i in range(len(split_steps) + 1)}
        for feature, entry in self.items():
            steps = entry['steps']
            for i, split_step in enumerate(split_steps):
                steps_i = [s for s in steps if s['model'] < split_step]
                steps = steps[len(steps_i) :]
                if any(steps_i):
                    if i > 0:
                        for s in steps_i:
                            s.update(
                                {'model': s['model'] - split_steps[i - 1]}
                            )
                    split_dict[i][feature] = {'steps': steps_i}
            if any(steps):
                for s in steps:
                    s.update({'model': s['model'] - split_steps[-1]})
                split_dict[len(split_steps)][feature] = {'steps': steps}

        return [ExoData(split) for split in split_dict.values()]

    def get_combine_type_data(self, feature, combine_type, model_step=None):
        """Get exogenous data for given feature which is used according to the
        given combine_type (input/output/layer) for this model_step.

        Parameters
        ----------
        feature : str
            Name of exogenous feature to get data for
        combine_type : str
            Usage type for requested data. e.g input/output/layer
        model_step : int | None
            Model step the data will be used for. If this is not None then
            only steps with self[feature]['steps'][:]['model'] == model_step
            will be searched for data.

        Returns
        -------
        data : tf.Tensor | np.ndarray
            Exogenous data for given parameters
        """
        tmp = self[feature]
        if model_step is not None:
            tmp = {k: v for k, v in tmp.items() if v['model'] == model_step}
        combine_types = [step['combine_type'] for step in tmp['steps']]
        msg = (
            'Received exogenous_data without any combine_type '
            f'= "{combine_type}" steps'
        )
        assert combine_type in combine_types, msg
        idx = combine_types.index(combine_type)
        return tmp['steps'][idx]['data']
