
# Standard Imports
import numpy as np
from typing import Any, Dict, List, Optional, Tuple,Union
from copy import deepcopy
from mne.io import Raw
from mne import Epochs

# Custom Imports
from .basePrimitive import PrimitiveFeatureMetadata, PrimitiveFeatureStructure
from .registry import PrimitiveFeatureRegistry
from .utils import get_mne_metadata

# Feature Imports
from mne.time_frequency import psd_array_multitaper, psd_array_welch

# Extra Imports
import matplotlib.pyplot as plt
import base64
from io import BytesIO


# enforce keyword only with *
def primitive_spectrum_feature(input: Union[Epochs,Raw],*, label: Optional[str] = None, method: str = "multitaper", mne_kwargs: Optional[Dict[str, Any]] = None) -> PrimitiveFeatureStructure:
    """
    Compute the power spectrum from time-domain EEG data using MNE's multitaper method.

    Parameters:
        input (Union[Epochs, Raw]): The input data to compute the power spectrum from.
        label (Optional[str]): A unique name for this instance. Default is None.
        method (str): The method to use for computing the power spectrum. Options are 'multitaper' or 'welch'.
                      Default is 'multitaper'.
        mne_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to pass to MNE functions.

    Returns:
        FeatureOutput: Contains power spectral density and metadata.
    """
    # Get metadata from the input data
    input_order, input_axes, extra_metadata = get_mne_metadata(input)

    # This is because both functions return the same axes and order but replaces time dimension at the end
    # I will use the fact that in python 3.7+ dicts are ordered
    # but order still saves the order of the dimensions in the array
    output_order = deepcopy(input_order)
    output_axes = deepcopy(input_axes)

    # time is the replaced dimension, and is the last one in the order
    time_idx = output_order.index('times')
    # remove time from the order
    output_order = list(output_order)
    output_order.remove('times')
    output_order = tuple(output_order)
    output_axes.pop('times', None) # remove time

    if method == "multitaper":
        output_kind = mne_kwargs.get('output', 'power') # default is power
        if output_kind == 'power':
            psds, freqs = psd_array_multitaper(input.get_data(), sfreq=input.info['sfreq'], **(mne_kwargs or {}))
            output_order = output_order + ('frequencies',)
            output_axes['frequencies'] = freqs
        elif output_kind == 'complex':
            psds, freqs, weights = psd_array_multitaper(input.get_data(), sfreq=input.info['sfreq'], **(mne_kwargs or {}))
            output_order = output_order + ('tapers', 'frequencies', )
            output_axes['tapers'] = list(range(weights.shape[0]))
            output_axes['frequencies'] = freqs
        else:
            raise ValueError("Output kind of Multitaper must be either 'power' or 'complex'.")

    elif method == "welch":
        average_kind = mne_kwargs.get('average', 'mean') # default is mean
        psds, freqs = psd_array_welch(input.get_data(), sfreq=input.info['sfreq'], **(mne_kwargs or {}))

        if average_kind in ['mean', 'median']:
            output_order = output_order + ('frequencies',)
            output_axes['frequencies'] = freqs
        elif average_kind is None:
            output_order = output_order + ('frequencies', 'segments',)
            output_axes['frequencies'] = freqs
            output_axes['segments'] = list(range(psds.shape[-1]))

    else:
        raise ValueError("Method must be either 'multitaper' or 'welch'.")


    # Create the PrimitiveFeatureMetadata object
    kwargs = dict(label=label, method=method, mne_kwargs=mne_kwargs)
    
    metadata = PrimitiveFeatureMetadata(
        label = label,
        kind = 'spectrum',
        type = 'array',
        axes = output_axes,
        order = output_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs
    )

    # Create the PrimitiveFeatureStructure object
    feature_structure = PrimitiveFeatureStructure(
        values = psds,
        metadata = metadata
    )

    return feature_structure

PrimitiveFeatureRegistry.register(primitive_spectrum_feature.__name__, "array", primitive_spectrum_feature)

# from ssqueezepy.experimental import scale_to_freq
# from ssqueezepy import Wavelet

# def scales_to_frequencies(scales, sampling_rate=1.0, w0=6.0):
#     """
#     Convert CWT scales to frequency equivalents for the Morlet wavelet.

#     Parameters:
#     - scales: Array-like, scales from the CWT.
#     - sampling_rate: float, sampling rate of the signal.
#     - w0: float, center frequency of the Morlet wavelet (usually around 6).

#     Returns:
#     - Array-like, frequency values corresponding to the given scales.
#     """
#     delta = 1.0 / sampling_rate
#     return w0 / (2 * np.pi * scales * delta)


"""
###TODO: Usually these methods will have baseline correction/normalization to make the high frequencies higher.
# We dont want that for foooof

raw, idxs_pos, idxs_neg = quickload(eegs[1])

sig  = raw.get_data()[idxs_pos[7]]
wavelet = Wavelet('morlet')
Wx, scales = cwt(sig, wavelet)
freqs_cwt = scale_to_freq(scales, wavelet, len(sig), fs=raw.info['sfreq'])#.astype(int)
imshow(Wx, abs=1,
       title="abs(CWT) | Morlet wavelet",
       ylabel="scales", xlabel="samples",yticks=freqs_cwt)
"""