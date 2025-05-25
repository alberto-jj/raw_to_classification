import numpy as np
from typing import Any, Dict, List, Optional, Tuple,Union
from copy import deepcopy
from scipy.stats import zscore
from mne.io import Raw
from mne import Epochs

from mne import make_fixed_length_epochs
from mne.datasets.eegbci import standardize
from mne.channels import make_standard_montage

# Custom Imports
from .base_functional import FunctionalFeatureMetadata, FunctionalFeatureStructure, FunctionalFeatureRegistry
from .decorators import functional_feature
from .utils import get_mne_metadata
from .utils import get_replaced_axes_order_values, get_sliced_index_combinations



@functional_feature('functional_cast_to_structure_feature', 'mne')
def functional_cast_to_structure_feature(
    input: Union[Raw, Epochs, FunctionalFeatureStructure],
    *,
    label: Optional[str] = None
) -> FunctionalFeatureStructure:
    """
    Wrap an MNE object (Raw or Epochs) into a FunctionalFeatureStructure if it's not already one.
    If the input is already a FunctionalFeatureStructure, it is returned as-is.
    """

    if isinstance(input, FunctionalFeatureStructure):
        return input  # Already structured

    # Otherwise assume it's Raw or Epochs
    input = input.copy()
    input_order, input_axes, extra_metadata = get_mne_metadata(input)
    kwargs = dict(label=label)

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind='mne',
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=kwargs,
    )

    return FunctionalFeatureStructure(
        values=input.copy(),
        metadata=metadata
    )

@functional_feature('functional_resample_feature', 'mne')
def functional_resample_feature(input: Union[Epochs,Raw,FunctionalFeatureStructure],*, label: Optional[str] = None, mne_kwargs: Dict[str, Any] = None) -> FunctionalFeatureStructure:
    """
    Resample the input data to a new sampling frequency.
    Can receive a FunctionalFeatureStructure holding an mne object or a Raw or Epochs object directly.
    """

    # Check if input is already a FunctionalFeatureStructure
    if isinstance(input, FunctionalFeatureStructure):
        # If it is, extract the values and metadata
        input = input.values

    # Get metadata
    input_order, input_axes, extra_metadata = get_mne_metadata(input)

    # Resample the data
    input = input.copy() # Create a copy to avoid modifying the original data
    resampled_data = input.resample(**(mne_kwargs if mne_kwargs else {}))
    input_order, input_axes, extra_metadata = get_mne_metadata(resampled_data)
    kwargs = dict(label=label, mne_kwargs=mne_kwargs)

    metadata = FunctionalFeatureMetadata(
        label = label,
        kind = 'mne',
        type_ = 'mne',
        axes = input_axes,
        order = input_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs,
    )

    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values = resampled_data.copy(),
        metadata = metadata
    )
    return feature_structure

@functional_feature('functional_make_fixed_length_epochs_feature', 'mne')
def functional_make_fixed_length_epochs_feature(input: Union[Epochs,FunctionalFeatureStructure],*, label: Optional[str] = None, mne_kwargs: Dict[str, Any] = None) -> FunctionalFeatureStructure:
    """
    Make fixed length epochs from the input data.
    Input must be Raw.
    """

    # Check if input is already a FunctionalFeatureStructure
    if isinstance(input, FunctionalFeatureStructure):
        # If it is, extract the values and metadata
        input = input.values


    input = input.copy() # Create a copy to avoid modifying the original data
    # Get metadata
    input_order, input_axes, extra_metadata = get_mne_metadata(input)

    # Resample the data
    input = input.copy() # Create a copy to avoid modifying the original data
    epoched_data = make_fixed_length_epochs(input, preload=True,**(mne_kwargs if mne_kwargs else {}))
    # The epochs constructor must preload the data for it to work in subsequent steps of the chain
    input_order, input_axes, extra_metadata = get_mne_metadata(epoched_data)
    kwargs = dict(label=label, mne_kwargs=mne_kwargs)

    metadata = FunctionalFeatureMetadata(
        label = label,
        kind = 'mne',
        type_ = 'mne',
        axes = input_axes,
        order = input_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs,
    )

    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values = epoched_data.copy(),
        metadata = metadata
    )
    return feature_structure

@functional_feature('functional_notch_filter_feature', 'mne')
def functional_notch_filter_feature(input: Union[Epochs,Raw,FunctionalFeatureStructure],*, label: Optional[str] = None, mne_kwargs: Dict[str, Any] = None) -> FunctionalFeatureStructure:
    """
    Apply a notch filter to the input data.
    Input must be Raw or Epochs.
    """

    # Check if input is already a FunctionalFeatureStructure
    if isinstance(input, FunctionalFeatureStructure):
        # If it is, extract the values and metadata
        input = input.values

    # Get metadata
    input_order, input_axes, extra_metadata = get_mne_metadata(input)

    # Notch filter
    input = input.copy() # Create a copy to avoid modifying the original data
    input = input.notch_filter(**(mne_kwargs if mne_kwargs else {}))
    input_order, input_axes, extra_metadata = get_mne_metadata(input)
    kwargs = dict(label=label, mne_kwargs=mne_kwargs)
    metadata = FunctionalFeatureMetadata(
        label = label,
        kind = 'mne',
        type_ = 'mne',
        axes = input_axes,
        order = input_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs,
    )
    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values = input.copy(),
        metadata = metadata
    )
    return feature_structure


@functional_feature('functional_channel_zscore_feature', 'mne')
def functional_channel_zscore_feature(input: Union[Epochs, Raw, FunctionalFeatureStructure], *, label: Optional[str] = None) -> FunctionalFeatureStructure:
    """
    Apply a z-score to the input data.
    Input must be Raw or Epochs.

    - For Raw: z-score is applied per channel across time.
    - For Epochs: z-score is applied globally across all epochs and time, per channel.
    """

    # Check if input is already a FunctionalFeatureStructure
    if isinstance(input, FunctionalFeatureStructure):
        # If it is, extract the values and metadata
        input = input.values

    # Get metadata
    input_order, input_axes, extra_metadata = get_mne_metadata(input)
    input = input.copy()  # avoid modifying original

    if len(input_order) == 2 and 'times' in input_axes:  # likely Raw: shape (n_channels, n_times)
        data = input.get_data()  # shape: (n_channels, n_times)
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        z_data = (data - mean) / std
        input._data = z_data

    elif len(input_order) == 3 and 'epochs' in input_axes and 'times' in input_axes:  # likely Epochs: shape (n_epochs, n_channels, n_times)
        data = input.get_data()  # shape: (n_epochs, n_channels, n_times)
        mean = data.mean(axis=(0, 2), keepdims=True)  # shape: (1, n_channels, 1)
        std = data.std(axis=(0, 2), keepdims=True)
        z_data = (data - mean) / std
        input._data = z_data
    else:
        raise ValueError("Input must be either mne.io.Raw or mne.Epochs with expected dimension order.")

    # Update metadata after transformation
    input_order, input_axes, extra_metadata = get_mne_metadata(input)
    kwargs = dict(label=label)
    metadata = FunctionalFeatureMetadata(
        label=label,
        kind='mne',
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=kwargs,
    )

    return FunctionalFeatureStructure(
        values=input.copy(),
        metadata=metadata
    )

@functional_feature('functional_filter_feature', 'mne')
def functional_filter_feature(input: Union[Epochs, Raw, FunctionalFeatureStructure], *, label: Optional[str] = None, mne_kwargs: Dict[str, Any] = None) -> FunctionalFeatureStructure:
    """
    Apply a bandpass filter to the input data.
    Input must be Raw or Epochs.
    """


    # Check if input is already a FunctionalFeatureStructure
    if isinstance(input, FunctionalFeatureStructure):
        # If it is, extract the values and metadata
        input = input.values

    # Get metadata
    input_order, input_axes, extra_metadata = get_mne_metadata(input)

    # Filter the data
    input = input.copy()  # Create a copy to avoid modifying the original data
    input = input.filter(**(mne_kwargs if mne_kwargs else {}))
    input_order, input_axes, extra_metadata = get_mne_metadata(input)
    kwargs = dict(label=label, mne_kwargs=mne_kwargs)

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind='mne',
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=kwargs,
    )

    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values=input.copy(),
        metadata=metadata
    )
    return feature_structure



@functional_feature('functional_standardize_channel_names_feature', 'mne')
def functional_standardize_channel_names_feature(
    input: Union[Raw, Epochs, FunctionalFeatureStructure],
    *,
    label: Optional[str] = None,
    keep_chans: Optional[List[str]] = None
) -> FunctionalFeatureStructure:
    """
    Optionally reorder channels and standardize channel names using mne.datasets.eegbci.standardize.

    Parameters:
        input: Raw or Epochs object
        label: Optional label
        keep_chans: If provided, reorder input to include only these channels (in order)

    Returns:
        FunctionalFeatureStructure containing updated MNE object
    """

    # Check if input is already a FunctionalFeatureStructure
    if isinstance(input, FunctionalFeatureStructure):
        # If it is, extract the values and metadata
        input = input.values

    input = input.copy()

    if keep_chans is not None:
        input.reorder_channels(keep_chans)

    # Standardize channel names in-place
    standardize(input)

    # Extract updated metadata
    input_order, input_axes, extra_metadata = get_mne_metadata(input)

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind='mne',
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=dict(label=label, keep_chans=keep_chans),
    )

    return FunctionalFeatureStructure(
        values=input.copy(),
        metadata=metadata
    )


@functional_feature('functional_set_montage_feature', 'mne')
def functional_set_montage_feature(
    input: Union[Raw, Epochs, FunctionalFeatureStructure],
    *,
    label: Optional[str] = None,
    mne_kwargs: Optional[Dict[str, Any]] = None
) -> FunctionalFeatureStructure:
    """
    Set an MNE standard montage to the input object using mne.channels.make_standard_montage.

    Parameters:
        input: Raw or Epochs object
        label: Optional label for the output
        mne_kwargs: Must include 'montage_kind' (e.g., 'standard_1005', 'biosemi64')

    Returns:
        FunctionalFeatureStructure with the montage set
    """

    # Check if input is already a FunctionalFeatureStructure
    if isinstance(input, FunctionalFeatureStructure):
        # If it is, extract the values and metadata
        input = input.values

    input = input.copy()
    mne_kwargs = mne_kwargs or {}

    montage_kind = mne_kwargs.get("montage_kind", None)
    if not montage_kind:
        raise ValueError("Missing required `montage_kind` in mne_kwargs.")

    montage = make_standard_montage(montage_kind)
    input = input.set_montage(montage)

    # Extract metadata after montage is applied
    input_order, input_axes, extra_metadata = get_mne_metadata(input)

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind='mne',
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=dict(label=label, **mne_kwargs),
    )

    return FunctionalFeatureStructure(
        values=input.copy(),
        metadata=metadata
    )
