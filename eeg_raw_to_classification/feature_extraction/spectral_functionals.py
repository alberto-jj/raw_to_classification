
# Standard Imports
import numpy as np
from typing import Any, Dict, List, Optional, Tuple,Union
from copy import deepcopy
from mne.io import BaseRaw
from mne import BaseEpochs
from inspect import currentframe

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


# enforce keyword only with *
@functional_feature_decorator('functional_spectrum_feature', 'array')
def functional_spectrum_feature(input: Union[BaseEpochs,BaseRaw],*, label: Optional[str] = None, method: str = "multitaper", mne_kwargs: Optional[Dict[str, Any]] = None) -> FunctionalFeatureStructure:
    """
    Compute the power spectrum from time-domain EEG data using MNE's multitaper method.

    Parameters:
        input (Union[BaseEpochs, BaseRaw]): The input data to compute the power spectrum from.
        label (Optional[str]): A unique name for this instance. Default is None.
        method (str): The method to use for computing the power spectrum. Options are 'multitaper' or 'welch'.
                      Default is 'multitaper'.
        mne_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to pass to MNE functions.

    Returns:
        FeatureOutput: Contains power spectral density and metadata.

    Note:
        if you want to use a pure array, you can use the mne.io.RawArray or mne.EpochsArray
        to create a BaseRaw or BaseEpochs object from a numpy array. This is useful for testing purposes.

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

    input = functional_cast_to_structure_feature(input, label=label)
    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    # Get metadata
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input)


    # In general for all features, validate the input array dimensions (number and order or set of dimensions)
    assert input.values.get_data().ndim in [2,3], "Input data must be 2D or 3D (e.g., mne.io.BaseRaw or mne.BaseEpochs)."


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
            psds, freqs = psd_array_multitaper(input.values.get_data(), sfreq=input.values.info['sfreq'], **(mne_kwargs or {}))
            output_order = output_order + ('frequencies',)
            output_axes['frequencies'] = freqs
        elif output_kind == 'complex':
            psds, freqs, weights = psd_array_multitaper(input.values.get_data(), sfreq=input.values.info['sfreq'], **(mne_kwargs or {}))
            output_order = output_order + ('tapers', 'frequencies', )
            output_axes['tapers'] = list(range(weights.shape[0]))
            output_axes['frequencies'] = freqs
        else:
            raise ValueError("Output kind of Multitaper must be either 'power' or 'complex'.")

    elif method == "welch":
        average_kind = mne_kwargs.get('average', 'mean') # default is mean
        psds, freqs = psd_array_welch(input.values.get_data(), sfreq=input.values.info['sfreq'], **(mne_kwargs or {}))

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
        kind = kind,
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

    # Update provenance
    feature_structure = update_provenance(input, feature_structure)

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

@functional_feature_decorator('functional_bandspectrum_feature', 'array')
def functional_bandspectrum_feature(input: FunctionalFeatureStructure, *, bands: Dict[str, Tuple[float, float]], relative: bool = False, label: Optional[str] = None) -> FunctionalFeatureStructure:
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
    band_list = list(BANDS.keys()) # important , if you do only .keys() it wont be pickable 
    new_order, new_axes,new_values = get_replaced_axes_order_values(axes, order, sliced_axis, 'bands', band_list)

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

    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind


    # Create the FunctionalFeatureMetadata object
    kwargs = dict(label=label, bands_dict=BANDS, relative=relative)
    metadata = FunctionalFeatureMetadata(
        label = label,
        kind = kind,
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


    # Update provenance
    feature_structure = update_provenance(input, feature_structure)


    return feature_structure

def single_fooof(freqs, psds, internal_kwargs: Dict[str, Dict[str, Any]]) -> FOOOF:
    """Fit a single FOOOF model with evaluated kwargs if needed."""
    kwargs = deepcopy(internal_kwargs)
    for section in kwargs:
        for key, value in kwargs[section].items():
            if isinstance(value, str) and 'eval%' in value:
                expression = value.replace('eval%', '')
                kwargs[section][key] = eval(expression)

    fm = FOOOF(verbose=False, **kwargs.get('FOOOF', {}))
    fm.fit(freqs, psds, **kwargs.get('fit', {}))
    return fm


@functional_feature_decorator('functional_fooof_feature', 'object')
def functional_fooof_feature(
    input: FunctionalFeatureStructure,
    *,
    internal_kwargs: Dict[str, Dict[str, Any]],
    label: Optional[str] = None
) -> FunctionalFeatureStructure:
    """
    Apply FOOOF to each 1D spectrum in the input FunctionalFeatureStructure.

    Parameters:
        input: FunctionalFeatureStructure with a 'frequencies' axis.
        internal_kwargs: Dictionary of kwargs to pass to FOOOF and its `fit` method.
        label: Optional label for metadata.

    Returns:
        FunctionalFeatureStructure containing FOOOF model objects.
    """
    assert 'frequencies' in input.metadata.axes, "'frequencies' must be in axes to apply FOOOF"
    freqs = input.metadata.axes['frequencies']
    spectrum = input.values

    axes = input.metadata.axes
    order = input.metadata.order
    sliced_axis = 'frequencies'

    # 👇 Use reduced axis logic instead of replaced
    new_order, new_axes, new_values = get_reduced_axes_order_values(axes, order, removed_axis=sliced_axis)

    # Determine index combinations that slice across all axes except 'frequencies'
    index_combinations = get_sliced_index_combinations(axes, order, sliced_axis)

    for this_idx in index_combinations:
        idx = [item[1] for item in this_idx]
        this_spectrum = spectrum[tuple(idx)]
        assert this_spectrum.shape == (len(freqs),), "Expected 1D spectrum for each slice."

        model = single_fooof(freqs, this_spectrum, internal_kwargs)

        # Drop frequency axis index — slice returns 1 model per slice
        new_idx = tuple(idx[i] for i, k in enumerate(order) if k != sliced_axis)
        new_values[new_idx] = model

    # Validate consistency
    for ax, items in new_axes.items():
        ax_index = new_order.index(ax)
        assert len(items) == new_values.shape[ax_index], \
            f"Shape mismatch for axis {ax}: {len(items)} != {new_values.shape[ax_index]}"

    extra_metadata = {
        'freqs': freqs,
        'removed_axis': sliced_axis,
        'fit_strategy': 'per-slice',
    }

    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind=kind,
        type_='array',
        axes=new_axes,
        order=new_order,
        extra_metadata=extra_metadata,
        kwargs=dict(label=label, internal_kwargs=internal_kwargs)
    )


    feature_structure = FunctionalFeatureStructure(
        values=new_values,
        metadata=metadata
    )
    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
    return feature_structure


@functional_feature_decorator('functional_fooof_component_feature', 'array')
def functional_fooof_component_feature(
    input: FunctionalFeatureStructure,
    *,
    component: str = "oscillatory",  # "original", "aperiodic", or "oscillatory"
    label: Optional[str] = None
) -> FunctionalFeatureStructure:
    """
    Extract specific spectral component from FOOOF models on a linear scale.

    Parameters:
        input: FunctionalFeatureStructure containing FOOOF model objects.
        component: Which part to extract:
            - "original": 10^FOOOF power_spectrum
            - "aperiodic": 10^FOOOF._ap_fit
            - "oscillatory": difference of the above
        label: Optional label for the output.

    Returns:
        FunctionalFeatureStructure of the selected spectrum (type='array'), with 'frequencies' axis added.
    """
    assert isinstance(input.values.flat[0], FOOOF), "Input values must contain FOOOF models"
    assert component in {"original", "aperiodic", "oscillatory"}, \
        f"Invalid component '{component}', must be one of: original, aperiodic, oscillatory"

    axes = deepcopy(input.metadata.axes)
    order = input.metadata.order
    sliced_axis = 'frequencies'

    # Recover frequency axis from the first FOOOF model
    sample_fooof: FOOOF = input.values.flat[0]
    freqs = sample_fooof.freqs
    axes[sliced_axis] = freqs
    new_order = list(order) + [sliced_axis]

    # Allocate output array
    shape = [len(axes[k]) for k in new_order]
    new_values = np.empty(shape, dtype=np.float32)

    # Iterate using structured slicing
    index_combinations = get_sliced_index_combinations(axes=input.metadata.axes, order=order, sliced_axis=sliced_axis)

    for combo in index_combinations:
        _, idx_nums = zip(*combo)
        fm: FOOOF = input.values[idx_nums]

        if component == "original":
            out = np.power(10, fm.power_spectrum)
        elif component == "aperiodic":
            out = np.power(10, fm._ap_fit)
        elif component == "oscillatory":
            out = np.power(10, fm.power_spectrum) - np.power(10, fm._ap_fit)

        new_idx = tuple(idx_nums) + (slice(None),)
        new_values[new_idx] = out

    kind = component
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind=kind,
        type_='array',
        axes=axes,
        order=tuple(new_order),
        extra_metadata={
            'retrieved_freqs': freqs.tolist(),
            'component': component
        },
        kwargs=dict(label=label, component=component)
    )

    feature_structure = FunctionalFeatureStructure(
        values=new_values,
        metadata=metadata
    )
    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
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