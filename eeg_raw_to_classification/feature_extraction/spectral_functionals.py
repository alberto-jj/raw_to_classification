
# Standard Imports
import numpy as np
from typing import Any, Dict, List, Optional, Tuple,Union
from copy import deepcopy
from mne.io import Raw
from mne import Epochs


# Custom Imports
from .base_functional import FunctionalFeatureMetadata, FunctionalFeatureStructure, FunctionalFeatureRegistry
from .decorators import functional_feature
from .utils import get_mne_metadata
from .utils import get_replaced_axes_order_values, get_sliced_index_combinations

# Feature Imports
from mne.time_frequency import psd_array_multitaper, psd_array_welch
from scipy.integrate import simpson as simps

# Extra Imports
import matplotlib.pyplot as plt
import base64
from io import BytesIO


# enforce keyword only with *
@functional_feature('functional_spectrum_feature', 'array')
def functional_spectrum_feature(input: Union[Epochs,Raw],*, label: Optional[str] = None, method: str = "multitaper", mne_kwargs: Optional[Dict[str, Any]] = None) -> FunctionalFeatureStructure:
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

    Note:
        if you want to use a pure array, you can use the mne.io.RawArray or mne.EpochsArray
        to create a Raw or Epochs object from a numpy array. This is useful for testing purposes.

    Example:
        times = np.linspace(0, 1, sampling_freq, endpoint=False)
        sine = np.sin(20 * np.pi * times)
        cosine = np.cos(10 * np.pi * times)
        shape = (n_channels, n_samples)
        data = np.array([sine, cosine])

        info = mne.create_info(
            ch_names=["10 Hz sine", "5 Hz cosine"], ch_types=["eeg"] * 2, sfreq=sampling_freq
        )

        simulated_raw = mne.io.RawArray(data, info)

        shape =(n_epochs, n_channels, n_samples)
        data = np.array(
            [
                [0.2 * sine, 1.0 * cosine],
                [0.4 * sine, 0.8 * cosine],
                [0.6 * sine, 0.6 * cosine],
                [0.8 * sine, 0.4 * cosine],
                [1.0 * sine, 0.2 * cosine],
            ]
        )

        simulated_epochs = mne.EpochsArray(data, info)
    """

    # Check if input is already a FunctionalFeatureStructure
    if isinstance(input, FunctionalFeatureStructure):
        # If it is, extract the values and metadata
        input = input.values

    # In general for all features, validate the input array dimensions (number and order or set of dimensions)
    assert input.get_data().ndim in [2,3], "Input data must be 2D or 3D (e.g., mne.io.Raw or mne.Epochs)."
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


    # Create the FunctionalFeatureMetadata object
    kwargs = dict(label=label, method=method, mne_kwargs=mne_kwargs)
    
    metadata = FunctionalFeatureMetadata(
        label = label,
        kind = 'Spectrum',
        type_ = 'array',
        axes = output_axes,
        order = output_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs
    )

    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values = psds,
        metadata = metadata
    )

    return feature_structure


def single_band_power(psd, freqs, band, relative=False):
    band = np.asarray(band)
    low, high = band
    if high is None:
        high = freqs[-1]  # Use the last frequency if no upper limit is specified
    if low is None:
        low = freqs[0]

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    # note, the way to aggregate the power could be different, e.g. passed as a parameter
    # but for now, we will use simpson
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

@functional_feature('functional_bandspectrum_feature', 'array')
def functional_bandspectrum_feature(input: FunctionalFeatureStructure, *, bands: Dict[str, Tuple[float, float]], relative: bool = False, label: Optional[str]) -> FunctionalFeatureStructure:
    """
    Compute the band power spectrum from Spectral data (e.g. frequencies is one of the axes).

    Parameters:
        input (FunctionalFeatureStructure): The input data to compute the band power spectrum from. 'frequencies' must be one of the axes.
        bands_dict (Dict[str, Tuple[float, float]]): A dictionary of frequency bands to compute the band power for.
        relative (bool): If True, return relative band power. Default is False.
        label (Optional[str]): A unique name for this instance. Default is None.

    Returns:
        FunctionalFeatureStructure: Contains band power spectral density and metadata.
    """

    BANDS = bands

    spectrum = input.values
    frequencies = input.metadata.axes['frequencies']

    axes = input.metadata.axes
    order = input.metadata.order

    sliced_axis = 'frequencies'

    new_order, new_axes,new_values = get_replaced_axes_order_values(axes, order, sliced_axis, 'bands', BANDS.keys())

    new_values.shape

    index_combinations = get_sliced_index_combinations(axes, order, sliced_axis)

    for band in BANDS.keys():
        for this_idx in index_combinations:
            items = [item[0] for item in this_idx]
            idx = [item[1] for item in this_idx]

            this_spectrum = spectrum[tuple(idx)]
            assert this_spectrum.shape == (len(frequencies),)  # should be a 1D array of frequencies
            band_power = single_band_power(this_spectrum, frequencies, BANDS[band], relative=relative)

            new_idx = list(idx)
            band_idx = list(BANDS.keys()).index(band)
            new_idx[order.index(sliced_axis)] = band_idx
            new_idx = tuple(new_idx)
            new_values[new_idx] = band_power

    for ax,items in new_axes.items():
        ax_index = new_order.index(ax)
        assert len(items) == new_values.shape[ax_index], f"Shape mismatch for axis {ax}: {len(items)} != {new_values.shape[ax_index]}"

    extra_metadata = {}
    extra_metadata['provenance'] = deepcopy(input.metadata)


    # Create the FunctionalFeatureMetadata object
    kwargs = dict(label=label, bands_dict=BANDS, relative=relative)
    metadata = FunctionalFeatureMetadata(
        label = label,
        kind = 'BandSpectrum',
        type_ = 'array',
        axes = new_axes,
        order = new_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs
    )

    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values = new_values,
        metadata = metadata
    )
    return feature_structure

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