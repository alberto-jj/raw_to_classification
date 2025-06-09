# Standard Imports
import numpy as np
from typing import Any, Dict, List, Optional, Tuple,Union, NamedTuple
from types import SimpleNamespace
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



def feature_factory(label, fun, package='Unknown', returns=None):
    @functional_feature_decorator(f'functional_{package}_{label}_feature', 'array')
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
                if isinstance(val, tuple) and fun.__name__ == 'hjorth_params' and package == 'antropy':
                    mobility, complexity = val
                    val = NamedTuple('hjorth_params')
                    val.mobility = mobility
                    val.complexity = complexity
                elif isinstance(val, tuple) and returns is not None:
                    # If returns are specified, ensure we only keep those
                    obj = SimpleNamespace(**{k: v for k, v in zip(returns, val)})
                    val = obj
                elif isinstance(val, tuple):
                    # encapsulate tuple in object for consistency
                    val = {i: v for i, v in enumerate(val)}
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
            extra_metadata={
                'package': package if package is not None else 'Unknown',
            },  # you can add more if you like
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


import neurokit2


complexity_list_antropy = [
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

complexity_measures_antropy = {x: getattr(antropy, x) for x in complexity_list_antropy}
generated_features = {}
for label, fun in complexity_measures_antropy.items():
    generated_features[label] = feature_factory(label, fun, package='antropy')
    # FunctionalFeatureRegistry.register(label, 'array', generated_features[label]) # already registered by decorator


from neurokit2.complexity import __all__ as complexity_nk2_list

skip_neurokit="""complexity_delay(signal, delay_max=50, method='fraser1986', algorithm=None, show=False, silent=False, **kwargs) --> float, info
complexity_dimension(signal, delay=1, dimension_max=20, method='afnn', show=False, **kwargs) --> float, info
complexity_tolerance(signal, method='maxApEn', r_range=None, delay=None, dimension=None, show=False) --> float, info
complexity_k(signal, k_max='max', show=False) --> float, info
mutual_information(x, y, method='varoquaux', bins='default', **kwargs) --> float
entropy_shannon_joint(x, y, base=2) --> float, info
fractal_tmf(signal, n=40, show=False, **kwargs) --> float, info
"""
# fractal_tmf crashes: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
# mutual_information(x, y, method='varoquaux', bins='default', **kwargs) needs two signals, not one
# entropy_shannon_joint(x, y, base=2) needs two signals, not one

neurokit2_signatures= """fractal_katz(signal) --> float, info
fractal_linelength(signal) --> float, info
fractal_petrosian(signal, symbolize='C', show=False) --> float, info
fractal_sevcik(signal) -->  float, info
fractal_nld(signal, corrected=False) --> dataframe, info
fractal_psdslope(signal, method='voss1988', show=False, **kwargs) --> float, info
fractal_higuchi(signal, k_max='default', show=False, **kwargs) --> float, info
fractal_density(signal, delay=1, tolerance='sd', bins=None, show=False, **kwargs) --> float, info
fractal_hurst(signal, scale='default', corrected=True, show=False) --> float, info
fractal_correlation(signal, delay=1, dimension=2, radius=64, show=False, **kwargs) --> float, info
fractal_dfa(signal, scale='default', overlap=True, integrate=True, order=1, multifractal=False, q='default', maxdfa=False, show=False, **kwargs) --> float|dataframe, info
entropy_shannon(signal=None, base=2, symbolize=None, show=False, freq=None, **kwargs) --> float, info
entropy_maximum(signal) --> float, info
entropy_differential(signal, base=2, **kwargs) --> float, info
entropy_power(signal, **kwargs) --> float, info
entropy_tsallis(signal=None, q=1, symbolize=None, show=False, freq=None, **kwargs) --> float, info
entropy_renyi(signal=None, alpha=1, symbolize=None, show=False, freq=None, **kwargs) --> float, info
entropy_approximate(signal, delay=1, dimension=2, tolerance='sd', corrected=False, **kwargs)  --> float, info
entropy_sample(signal, delay=1, dimension=2, tolerance='sd', **kwargs) --> float, info
entropy_quadratic(signal, delay=1, dimension=2, tolerance='sd', **kwargs) --> float, info
entropy_cumulativeresidual(signal, symbolize=None, show=False, freq=None) --> float, info
entropy_rate(signal, kmax=10, symbolize='mean', show=False)  --> float, info
entropy_svd(signal, delay=1, dimension=2, show=False)  --> float, info
entropy_kl(signal, delay=1, dimension=2, norm='euclidean', **kwargs) --> float, info
entropy_spectral(signal, bins=None, show=False, **kwargs) --> float, info
entropy_phase(signal, delay=1, k=4, show=False, **kwargs) --> float, info
entropy_grid(signal, delay=1, k=3, show=False, **kwargs) --> float, info
entropy_attention(signal, show=False, silent=False, **kwargs)  --> float, info, kwargs
entropy_increment(signal, dimension=2, q=4, **kwargs) --> float, info
entropy_slope(signal, dimension=3, thresholds=[0.1, 45], **kwargs) --> float, info
entropy_symbolicdynamic(signal, dimension=3, symbolize='MEP', c=6, **kwargs) --> float, info
entropy_dispersion(signal, delay=1, dimension=3, c=6, symbolize='NCDF', fluctuation=False, rho=1, **kwargs) --> float, info
entropy_ofentropy(signal, scale=10, bins=10, **kwargs) --> float, info
entropy_permutation(signal, delay=1, dimension=3, corrected=True, weighted=False, conditional=False, **kwargs) --> float, info
entropy_bubble(signal, delay=1, dimension=3, alpha=2, **kwargs) --> float, info
entropy_range(signal, dimension=3, delay=1, tolerance='sd', approximate=False, **kwargs) --> float, info
entropy_fuzzy(signal, delay=1, dimension=2, tolerance='sd', approximate=False, **kwargs) --> float, info
entropy_multiscale(signal, scale='default', dimension=3, tolerance='sd', method='MSEn', show=False, **kwargs) --> float, info
entropy_hierarchical(signal, scale='default', dimension=2, tolerance='sd', show=False, **kwargs) --> float, info
fisher_information(signal, delay=1, dimension=2) --> float, info
fishershannon_information(signal, **kwargs) --> float, info
complexity_hjorth(signal) --> float, info
complexity_decorrelation(signal, show=False) --> float, info
complexity_lempelziv(signal, delay=1, dimension=2, permutation=False, symbolize='mean', **kwargs) --> float, info
complexity_relativeroughness(signal, **kwargs) --> float, info
complexity_lyapunov(signal, delay=1, dimension=2, method='rosenstein1993', separation='auto', **kwargs) --> float, info
complexity_rqa(signal, dimension=3, delay=1, tolerance='sd', min_linelength=2, method='python', show=False) --> rqa dataframe, info
complexity_embedding(signal, delay=1, dimension=3, show=False, **kwargs) --> embedding array
complexity_symbolize(signal, method='mean', c=3, random_state=None, show=False, **kwargs) --> symbolized signal array
complexity_coarsegraining(signal, scale=2, method='nonoverlapping', show=False, **kwargs) --> coarsegrained signal array
complexity_ordinalpatterns(signal, delay=1, dimension=3, algorithm='quicksort', **kwargs) --> ordinal patterns, frequency array, info
recurrence_matrix(signal, delay=1, dimension=3, tolerance='default', show=False) --> recurrence array, dist. array"""

for line in neurokit2_signatures.split('\n'):
    label = line.split('(')[0]
    arguments = '('+line.split('(')[1].split(')')[0] + ')'
    arguments = arguments.replace('**', '')
    list_args = arguments.replace('(', '').replace(')', '').split(',')
    #list_args = [x+'="no-default"' if '=' not in x else x for x in list_args ]
    list_args = [x if '=' not in x else x.split('=')[0] for x in list_args]
    list_args = ','.join(list_args)
    returns = line.split('->')[1].strip()
    returns = returns.split(',') if ',' in returns else [returns]
    returns = [x.strip().replace(' ','') for x in returns]
    returns = [f'{x}' for x in returns]
    #print(label, list_args, returns, end=' | ')
    if set(returns)== {'float', 'info'}:
        #print('will be defined')
        fun = getattr(neurokit2, label)
        generated_features[label] = feature_factory(label, fun, package='neurokit2', returns=returns)
    else:
        #print('will not be defined, returns are not float and info')
        pass



ordpy_list = [
    'permutation_entropy(data, dx=3, dy=1, taux=1, tauy=1, base=2, normalized=True, probs=False, tie_precision=None) --> float' ,
    'complexity_entropy(data, dx=3, dy=1, taux=1, tauy=1, probs=False, tie_precision=None) --> perm_entropy, stat_complexity',
    'tsallis_entropy(data, q=1, dx=3, dy=1, taux=1, tauy=1, probs=False, tie_precision=None) --> float|array',
    ...
]

#TODO: add ordpy features
