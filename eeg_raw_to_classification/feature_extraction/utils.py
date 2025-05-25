from copy import deepcopy
import numpy as np
import re

from copy import deepcopy
import warnings
from typing import Tuple

from copy import deepcopy
import warnings

from copy import deepcopy
import warnings

from copy import deepcopy
from mne.io import BaseRaw
from mne import BaseEpochs
from typing import Union, Tuple, Dict, Any, List
from itertools import product


def update_provenance(input_struct, output_struct):
    """
    Returns a deepcopy of output_struct with updated top-level provenance,
    appending a clean copy of input_struct.metadata (without its own provenance).

    Parameters:
        input_struct (FunctionalFeatureStructure): The source feature.
        output_struct (FunctionalFeatureStructure): The derived feature.

    Returns:
        FunctionalFeatureStructure: A copy of output_struct with provenance updated.
    """
    output_copy = deepcopy(output_struct)

    # Clean input metadata (exclude its own provenance)
    prev_metadata = deepcopy(input_struct.metadata)
    prev_provenance = deepcopy(prev_metadata.provenance)

    if not isinstance(prev_provenance, list):
        warnings.warn(
            f"'provenance' is of type {type(prev_provenance).__name__}, not list. Resetting it."
        )
        prev_provenance = [prev_provenance]
    prev_metadata.provenance = ['Emptied']

    # Ensure output provenance is a list
    if output_copy.metadata.provenance is None:
        output_copy.metadata.provenance = []
    elif not isinstance(output_copy.metadata.provenance, list):
        warnings.warn(
            f"'provenance' is of type {type(output_copy.metadata.provenance).__name__}, not list. Resetting it."
        )
        output_copy.metadata.provenance = ['Emptied']

    # Append input metadata to output's provenance
    output_copy.metadata.provenance = prev_provenance + [prev_metadata]

    return output_copy


def is_bids_like_filename(filename: str) -> bool:
    """
    Check if a filename matches a relaxed BIDS-like pattern:
    key1-value1_key2-value2_suffix.extension

    Rules:
    - Keys and suffix use alphanumeric only: [a-zA-Z0-9]+
    - Values: any characters except underscore `_` or dash `-`
    - Must end with a period + extension (e.g., .fif, .nii.gz)

    Returns True if it matches, False otherwise.
    """
    pattern = (
        r"^"                         # Start of string
        r"([a-zA-Z0-9]+-[^_^-]+)"    # First key-value pair (value cannot contain _ or -)
        r"(_[a-zA-Z0-9]+-[^_^-]+)*"  # Zero or more additional key-value pairs
        r"_([a-zA-Z0-9]+)"           # Suffix (alphanumeric only)
        r"(\.[a-zA-Z0-9]+)+$"        # Extension (e.g., .fif, .nii.gz)
    )
    return re.match(pattern, filename) is not None

def snake_to_camel(snake_str):
    """Convert a snake_case string to CamelCase.

    Parameters
    ----------
    snake_str : str
        The input string in snake_case format.

    Returns
    -------
    str
        The converted string in CamelCase format.
    """
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def get_kind_from_snake(snake):
    snake = snake.replace('functional_','')
    snake = snake.replace('_functional','')
    snake = snake.replace('_feature','')
    snake = snake.replace('feature_','')
    snake = snake.replace('chain_','')
    snake = snake.replace('_chain','')
    return snake_to_camel(snake)


def get_mne_metadata(
    input: Union[BaseRaw, BaseEpochs, Any],
    save_input_no_data: bool = False
) -> Tuple[Tuple[str, ...], Dict[str, Any], Dict[str, Any], List[Any]]:
    """
    Extract standardized metadata from a BaseRaw, BaseEpochs, or FunctionalFeatureStructure input.

    Parameters
    ----------
    input : BaseRaw, BaseEpochs, or FunctionalFeatureStructure
        The input object to extract metadata from.
    save_input_no_data : bool
        If True and input is BaseRaw or BaseEpochs, save a copy without data to reduce memory usage.

    Returns
    -------
    input_order : tuple
        Axis order.
    input_axes : dict
        Axis name → values.
    extra_metadata : dict
        Includes input type and optionally a no-data copy.
    provenance : list
        List of filenames or references.
    """
    input_axes = {}
    input_order = ()
    extra_metadata = {}
    provenance = []

    # Case 1: Already structured, extract components directly
    if hasattr(input, 'metadata') and hasattr(input, 'values'):
        input_order = input.metadata.order
        input_axes = input.metadata.axes
        extra_metadata = deepcopy(input.metadata.extra_metadata or {})
        provenance = deepcopy(input.metadata.provenance or [])
        return input_order, input_axes, extra_metadata, provenance

    # Case 2: MNE BaseRaw or BaseEpochs
    elif hasattr(input, 'get_data') and hasattr(input, 'times') and hasattr(input, 'ch_names'):
        shape = input.get_data().shape

        if len(shape) == 2:  # BaseRaw
            input_order = ('spaces', 'times')
            provenance = [deepcopy(input.filenames)]
            extra_metadata['input_type'] = 'Raw'
        elif len(shape) == 3:  # BaseEpochs
            input_order = ('epochs', 'spaces', 'times')
            input_axes['epochs'] = list(range(shape[0]))
            provenance = [deepcopy(input.filename)]
            extra_metadata['input_type'] = 'BaseEpochs'
        else:
            raise ValueError("Unsupported MNE input shape.")

        input_axes['spaces'] = deepcopy(input.ch_names)
        input_axes['times'] = deepcopy(input.times)

        # input_no_data.info has the 
        # montage : info.get_montage()
        # sfreq
        # ch_types : info.get_channel_types()
        # and other metadata
        # you can use data from input_no_data to get the metadata
        # only the data is not there to save space
        # obviusly, some methods will not work...


        if save_input_no_data:
            input_no_data = input.copy()
            input_no_data._data = None
            input_no_data.preload = False
            extra_metadata['input_no_data'] = input_no_data
        else:
            extra_metadata['input_no_data'] = None

        return input_order, input_axes, extra_metadata, provenance

    else:
        raise TypeError("Unsupported input type for get_mne_metadata. Must be Raw, BaseEpochs, or FunctionalFeatureStructure.")




def get_sliced_index_combinations(axes,order, sliced_axis):
    axes_with_slice = {k: ([None] if k == sliced_axis else v) for k, v in axes.items()}

    axes_with_slice_and_indexes ={}
    for ax,list_of_items in axes_with_slice.items():
        if ax != sliced_axis:
            item_idx = [(item, i) for i, item in enumerate(list_of_items)]
            axes_with_slice_and_indexes[ax] = item_idx
        else:
            axes_with_slice_and_indexes[ax] =[(ax,slice(None))]

    index_combinations = list(product(*[axes_with_slice_and_indexes[k] for k in order]))
    assert len(index_combinations) == np.prod([len(axes[k]) if k != sliced_axis else 1 for k in order])
    return index_combinations
def get_replaced_axes_order_values(axes, order,replaced_axis, new_axis, new_axis_items):

    
    new_order = list(deepcopy(order))
    idx_replaced = new_order.index(replaced_axis)  # get the index of frequencies in the order
    new_order[idx_replaced] = new_axis  # replace frequencies with bands
    new_axes = {}

    for k in order:
        if k == replaced_axis:
            new_axes[new_axis] = new_axis_items
        else:
            new_axes[k] = axes[k]  # copy the axes
    new_order = tuple(new_order)  # convert to tuple

    new_shape = []

    for k, v in new_axes.items():
        new_shape.append(len(v))

    new_values = np.empty(new_shape, dtype=object) # create an empty array with the new shape

    return new_order, new_axes, new_values

def get_reduced_axes_order_values(axes, order, removed_axis):
    """
    Remove an axis from the axes and order, and create an empty array
    with the resulting shape (all other axes retained).

    Parameters:
        axes: Dict[str, List[Any]] - original axes
        order: Tuple[str] - original axis order
        removed_axis: str - axis to remove

    Returns:
        new_order: Tuple[str]
        new_axes: Dict[str, List[Any]]
        new_values: np.ndarray (with shape matching remaining axes)
    """
    new_order = tuple(k for k in order if k != removed_axis)
    new_axes = {k: v for k, v in axes.items() if k != removed_axis}

    new_shape = [len(new_axes[k]) for k in new_order]
    new_values = np.empty(new_shape, dtype=object)  # object just in case of structured output

    return new_order, new_axes, new_values
