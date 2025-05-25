import numpy as np
from typing import Any, Dict, List, Optional, Tuple,Union
from copy import deepcopy
from scipy.stats import zscore
from mne.io import BaseRaw
from mne import BaseEpochs

from mne import make_fixed_length_epochs
from mne.datasets.eegbci import standardize
from mne.channels import make_standard_montage

# Custom Imports
from .base_functional import FunctionalFeatureMetadata, FunctionalFeatureStructure, FunctionalFeatureRegistry
from .decorators import functional_feature_decorator
from .utils import get_mne_metadata, update_provenance, get_kind_from_snake
from inspect import currentframe
from .utils import get_replaced_axes_order_values, get_sliced_index_combinations



@functional_feature_decorator('functional_cast_to_structure_feature', 'mne')
def functional_cast_to_structure_feature(
    input: Union[BaseRaw, BaseEpochs, FunctionalFeatureStructure],
    *,
    label: Optional[str] = None
) -> FunctionalFeatureStructure:
    """
    Wrap an MNE object (BaseRaw or BaseEpochs) into a FunctionalFeatureStructure if it's not already one.
    If the input is already a FunctionalFeatureStructure, it is returned as-is.
    """

    if isinstance(input, FunctionalFeatureStructure):
        return input  # Already structured

    # Otherwise assume it's BaseRaw or BaseEpochs
    input = input.copy()
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input)
    kwargs = dict(label=label)

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind='mne',
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=kwargs,
        provenance=provenance
    )

    return FunctionalFeatureStructure(
        values=input.copy(),
        metadata=metadata
    )

@functional_feature_decorator('functional_resample_feature', 'mne')
def functional_resample_feature(input: Union[BaseEpochs,BaseRaw,FunctionalFeatureStructure],*, label: Optional[str] = None, mne_kwargs: Dict[str, Any] = None) -> FunctionalFeatureStructure:
    """
    Resample the input data to a new sampling frequency.
    Can receive a FunctionalFeatureStructure holding an mne object or a BaseRaw or BaseEpochs object directly.
    """

    input = functional_cast_to_structure_feature(input, label=label)
    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    # Get metadata
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input)

    # Resample the data
    input_mne = input.values.copy() # Create a copy to avoid modifying the original data
    resampled_data = input_mne.resample(**(mne_kwargs if mne_kwargs else {}))
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(resampled_data)
    kwargs = dict(label=label, mne_kwargs=mne_kwargs)

    metadata = FunctionalFeatureMetadata(
        label = label,
        kind = kind,
        type_ = 'mne',
        axes = input_axes,
        order = input_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs,
        provenance = provenance
    )

    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values = resampled_data.copy(),
        metadata = metadata
    )

    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
    return feature_structure

@functional_feature_decorator('functional_make_fixed_length_epochs_feature', 'mne')
def functional_make_fixed_length_epochs_feature(input: Union[BaseEpochs,FunctionalFeatureStructure],*, label: Optional[str] = None, mne_kwargs: Dict[str, Any] = None) -> FunctionalFeatureStructure:
    """
    Make fixed length epochs from the input data.
    Input must be BaseRaw.
    """

    input = functional_cast_to_structure_feature(input, label=label)


    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind


    input_mne = input.values.copy() # Create a copy to avoid modifying the original data
    # Get metadata
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input)

    epoched_data = make_fixed_length_epochs(input_mne, preload=True,**(mne_kwargs if mne_kwargs else {}))
    # The epochs constructor must preload the data for it to work in subsequent steps of the chain
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(epoched_data)
    kwargs = dict(label=label, mne_kwargs=mne_kwargs)

    metadata = FunctionalFeatureMetadata(
        label = label,
        kind =  kind,
        type_ = 'mne',
        axes = input_axes,
        order = input_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs,
        provenance = provenance
    )

    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values = epoched_data.copy(),
        metadata = metadata
    )
    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
    return feature_structure

@functional_feature_decorator('functional_notch_filter_feature', 'mne')
def functional_notch_filter_feature(input: Union[BaseRaw,FunctionalFeatureStructure],*, label: Optional[str] = None, mne_kwargs: Dict[str, Any] = None) -> FunctionalFeatureStructure:
    """
    Apply a notch filter to the input data.
    Input must be BaseRaw. #no notch filter for epochs
    """

    input = functional_cast_to_structure_feature(input, label=label)

    if not isinstance(input.values, BaseRaw):
        raise ValueError("Input must be BaseRaw for notch filtering.")
    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    # Get metadata
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input)

    # Notch filter
    input_mne = input.values.copy() # Create a copy to avoid modifying the original data
    input_mne = input_mne.notch_filter(**(mne_kwargs if mne_kwargs else {}))
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input_mne)
    kwargs = dict(label=label, mne_kwargs=mne_kwargs)
    metadata = FunctionalFeatureMetadata(
        label = label,
        kind = kind,
        type_ = 'mne',
        axes = input_axes,
        order = input_order,
        extra_metadata = extra_metadata,
        kwargs = kwargs,
    )
    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values = input_mne.copy(),
        metadata = metadata
    )
    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
    return feature_structure


@functional_feature_decorator('functional_channel_zscore_feature', 'mne')
def functional_channel_zscore_feature(input: Union[BaseEpochs, BaseRaw, FunctionalFeatureStructure], *, label: Optional[str] = None) -> FunctionalFeatureStructure:
    """
    Apply a z-score to the input data.
    Input must be BaseRaw or BaseEpochs.

    - For BaseRaw: z-score is applied per channel across time.
    - For BaseEpochs: z-score is applied globally across all epochs and time, per channel.
    """

    input = functional_cast_to_structure_feature(input, label=label)
    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    # Get metadata
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input)
    input_mne = input.values.copy()  # avoid modifying original

    if len(input_order) == 2 and 'times' in input_axes:  # likely BaseRaw: shape (n_channels, n_times)
        data = input_mne.get_data()  # shape: (n_channels, n_times)
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        z_data = (data - mean) / std
        input_mne._data = z_data

    elif len(input_order) == 3 and 'epochs' in input_axes and 'times' in input_axes:  # likely BaseEpochs: shape (n_epochs, n_channels, n_times)
        data = input_mne.get_data()  # shape: (n_epochs, n_channels, n_times)
        mean = data.mean(axis=(0, 2), keepdims=True)  # shape: (1, n_channels, 1)
        std = data.std(axis=(0, 2), keepdims=True)
        z_data = (data - mean) / std
        input_mne._data = z_data
    else:
        raise ValueError("Input must be either mne.io.BaseRaw or mne.BaseEpochs with expected dimension order.")

    # Update metadata after transformation
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input_mne)
    kwargs = dict(label=label)
    metadata = FunctionalFeatureMetadata(
        label=label,
        kind=kind,
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=kwargs,
    )

    feature_structure = FunctionalFeatureStructure(
        values=input_mne.copy(),
        metadata=metadata
    )

    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
    return feature_structure

@functional_feature_decorator('functional_filter_feature', 'mne')
def functional_filter_feature(input: Union[BaseEpochs, BaseRaw, FunctionalFeatureStructure], *, label: Optional[str] = None, mne_kwargs: Dict[str, Any] = None) -> FunctionalFeatureStructure:
    """
    Apply a bandpass filter to the input data.
    Input must be BaseRaw or BaseEpochs.
    """

    input = functional_cast_to_structure_feature(input, label=label)


    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    # Get metadata
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input)

    # Filter the data
    input_mne = input.values.copy()  # Create a copy to avoid modifying the original data
    input_mne = input_mne.filter(**(mne_kwargs if mne_kwargs else {}))
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input_mne)
    kwargs = dict(label=label, mne_kwargs=mne_kwargs)

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind=kind,
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=kwargs,
    )

    # Create the FunctionalFeatureStructure object
    feature_structure = FunctionalFeatureStructure(
        values=input_mne.copy(),
        metadata=metadata
    )

    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
    return feature_structure



@functional_feature_decorator('functional_standardize_channel_names_feature', 'mne')
def functional_standardize_channel_names_feature(
    input: Union[BaseRaw, BaseEpochs, FunctionalFeatureStructure],
    *,
    label: Optional[str] = None,
    keep_chans: Optional[List[str]] = None
) -> FunctionalFeatureStructure:
    """
    Optionally reorder channels and standardize channel names using mne.datasets.eegbci.standardize.

    Parameters:
        input: BaseRaw or BaseEpochs object
        label: Optional label
        keep_chans: If provided, reorder input to include only these channels (in order)

    Returns:
        FunctionalFeatureStructure containing updated MNE object
    """

    input = functional_cast_to_structure_feature(input, label=label)


    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    input_mne= input.values.copy()

    if keep_chans is not None:
        input_mne.reorder_channels(keep_chans)

    # Standardize channel names in-place
    standardize(input_mne)

    # Extract updated metadata
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input_mne)

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind=kind,
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=dict(label=label, keep_chans=keep_chans),
    )

    feature_structure = FunctionalFeatureStructure(
        values=input_mne.copy(),
        metadata=metadata
    )
    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
    return feature_structure


@functional_feature_decorator('functional_set_montage_feature', 'mne')
def functional_set_montage_feature(
    input: Union[BaseRaw, BaseEpochs, FunctionalFeatureStructure],
    *,
    label: Optional[str] = None,
    mne_kwargs: Optional[Dict[str, Any]] = None
) -> FunctionalFeatureStructure:
    """
    Set an MNE standard montage to the input object using mne.channels.make_standard_montage.

    Parameters:
        input: BaseRaw or BaseEpochs object
        label: Optional label for the output
        mne_kwargs: Must include 'montage_kind' (e.g., 'standard_1005', 'biosemi64')

    Returns:
        FunctionalFeatureStructure with the montage set
    """

    input = functional_cast_to_structure_feature(input, label=label)


    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    input_mne = input.values.copy()
    mne_kwargs = mne_kwargs or {}

    montage_kind = mne_kwargs.get("montage_kind", None)
    if not montage_kind:
        raise ValueError("Missing required `montage_kind` in mne_kwargs.")

    montage = make_standard_montage(montage_kind)
    input_mne = input_mne.set_montage(montage)

    # Extract metadata after montage is applied
    input_order, input_axes, extra_metadata, provenance = get_mne_metadata(input_mne)

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind=kind,
        type_='mne',
        axes=input_axes,
        order=input_order,
        extra_metadata=extra_metadata,
        kwargs=dict(label=label, **mne_kwargs),
    )

    feature_structure = FunctionalFeatureStructure(
        values=input_mne.copy(),
        metadata=metadata
    )

    # Update provenance
    feature_structure = update_provenance(input, feature_structure)
    return feature_structure