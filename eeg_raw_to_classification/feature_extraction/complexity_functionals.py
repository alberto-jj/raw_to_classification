# Standard Imports
import numpy as np
from typing import Any, Dict, List, Optional, Tuple,Union, NamedTuple
from copy import deepcopy
from mne.io import BaseRaw
from mne import BaseEpochs
from inspect import currentframe
from inspect import signature

# Custom Imports
from .base_functional import FunctionalFeatureMetadata, FunctionalFeatureStructure, FunctionalFeatureRegistry
from .prep_functionals import functional_cast_to_structure_feature
from .decorators import functional_feature_decorator
from .utils import get_mne_metadata
from .utils import get_replaced_axes_order_values, get_sliced_index_combinations, get_reduced_axes_order_values
from .utils import get_kind_from_snake, update_provenance

# Feature Imports
from mne.time_frequency import psd_array_multitaper, psd_array_welch
from scipy.integrate import simpson as simps
from fooof import FOOOF

# Extra Imports, maybe useful once we implement feature inspection/visualization
# import matplotlib.pyplot as plt
# import base64
# from io import BytesIO



def feature_factory(label, fun):
    @functional_feature_decorator(f'functional_{label}_feature', 'array')
    def feature_func(input, *, label=None, **kwargs):
        # Cast input to your expected structure and fetch data
        input = functional_cast_to_structure_feature(input, label=label)
        sf = input.values.info['sfreq'] if isinstance(input, BaseRaw) or isinstance(input, BaseEpochs) else input.metadata.extra_metadata.get('sfreq', None)
        data = input.values.get_data()
        ch_names = input.values.info['ch_names']

        kind = get_kind_from_snake(fun.__name__)

        # Handle both Raw (2D) and Epochs (3D)
        if data.ndim == 2:  # (channels, time)
            n_epochs = 1
            n_channels = data.shape[0]
            data = data[None, :, :]  # (1, channels, time)
        elif data.ndim == 3:  # (epochs, channels, time)
            n_epochs = data.shape[0]
            n_channels = data.shape[1]
        else:
            raise ValueError("Input must be 2D or 3D array.")

        # Compute the entropy feature for each epoch and channel
        values = []
        for epoch in range(n_epochs):
            epoch_values = []
            for ch in range(n_channels):
                if 'sf' in signature(fun).parameters and not 'sf' in kwargs:
                    # If the function requires sfreq and it's not provided, add it
                    kwargs['sf'] = sf
                val = fun(data[epoch, ch, :], **kwargs)
                if isinstance(val, tuple) and fun.__name__ == 'hjorth_params':
                    mobility, complexity = val
                    val = NamedTuple('hjorth_params')
                    val.mobility = mobility
                    val.complexity = complexity

                epoch_values.append(val)
            values.append(epoch_values)
        values = np.array(values)

        # Prepare axes and order
        axes = {'epochs': list(range(n_epochs)), 'spaces': ch_names}
        order = ('epochs', 'spaces')
        if n_epochs == 1:
            # Squeeze epochs for raw
            values = values.squeeze(0)
            axes.pop('epochs')
            order = ('spaces',)

        # Prepare metadata
        metadata = FunctionalFeatureMetadata(
            label=label if label is not None else fun.__name__,
            kind=kind,
            type_='array',
            axes=axes,
            order=order,
            extra_metadata={},  # you can add more if you like
            kwargs=kwargs
        )

        # Build feature structure
        feature_structure = FunctionalFeatureStructure(
            values=values,
            metadata=metadata
        )

        # Optionally update provenance
        feature_structure = update_provenance(input, feature_structure)

        return feature_structure
    return feature_func

import antropy

from neurokit2 import entropy_multiscale

complexity_list = [
    'detrended_fluctuation',
    'lziv_complexity',
    'sample_entropy',
    'spectral_entropy',
    'app_entropy',
    'hjorth_params',
    'num_zerocross',
    'perm_entropy',
    'svd_entropy',
    'higuchi_fd',
    'katz_fd',
    'petrosian_fd',
]

complexity_measures = {x: getattr(antropy, x) for x in complexity_list}
generated_features = {}
for label, fun in complexity_measures.items():
    generated_features[label] = feature_factory(label, fun)
    # FunctionalFeatureRegistry.register(label, 'array', generated_features[label]) # already registered by decorator

