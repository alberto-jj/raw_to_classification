import numpy as np
from typing import Any, Dict, List, Optional, Tuple,Union, Callable
from copy import deepcopy

# Custom Imports
from .base_functional import FunctionalFeatureMetadata, FunctionalFeatureStructure, FunctionalFeatureRegistry
from .decorators import functional_feature_decorator

from itertools import permutations
from .utils import get_mne_metadata, update_provenance, get_kind_from_snake
from inspect import currentframe
from .utils import snake_to_camel



@functional_feature_decorator('functional_ratio_feature', 'array')
def functional_ratio_feature(
    input: FunctionalFeatureStructure,
    *,
    label: Optional[str] = None,
    ratio_axis: str = 'bands'
) -> FunctionalFeatureStructure:
    """
    Compute all pairwise ratios of elements along a specified axis.

    Parameters:
        input: FunctionalFeatureStructure with a numeric array.
        label: Optional label for the output.
        ratio_axis: Name of the axis to compute pairwise ratios over (e.g., "bands").

    Returns:
        FunctionalFeatureStructure with a new axis 'band_ratios' replacing the original axis.
    """
    assert ratio_axis in input.metadata.axes, f"'{ratio_axis}' must be in input axes"

    kind = get_kind_from_snake(currentframe().f_code.co_name)
    kind = input.metadata.kind + kind if input.metadata.kind else kind

    dimension = kind.lower() + "s"
    axes = input.metadata.axes
    order = input.metadata.order
    values = input.values

    items = axes[ratio_axis]
    item_pairs = list(permutations(items, 2))  # ordered pairs, exclude self-pairs

    # Create new axes/order with 'band_ratios' replacing the original axis
    new_order = list(order)
    idx = new_order.index(ratio_axis)
    new_order[idx] = dimension

    new_axes = deepcopy(axes)
    del new_axes[ratio_axis]
    new_axes[dimension] = item_pairs

    # Compute new shape and initialize
    new_shape = [len(new_axes[ax]) for ax in new_order]
    new_values = np.full(new_shape, np.nan, dtype=np.float32)

    # Build axis index combinations for slicing
    from .utils import get_sliced_index_combinations  # assumed present
    index_combinations = get_sliced_index_combinations(axes, order, ratio_axis)

    for (top_label, bottom_label) in item_pairs:
        top_idx = items.index(top_label)
        bottom_idx = items.index(bottom_label)
        ratio_idx = item_pairs.index((top_label, bottom_label))

        for combo in index_combinations:
            labels, idxs = zip(*combo)
            # Build full index for original data
            full_idx_top = list(idxs)
            full_idx_bottom = list(idxs)
            full_idx_top[order.index(ratio_axis)] = top_idx
            full_idx_bottom[order.index(ratio_axis)] = bottom_idx

            numerator = values[tuple(full_idx_top)]
            denominator = values[tuple(full_idx_bottom)]

            new_idx = list(idxs)
            new_idx[order.index(ratio_axis)] = ratio_idx
            new_idx = tuple(new_idx)

            if denominator == 0:
                new_values[new_idx] = np.nan
            else:
                new_values[new_idx] = numerator / denominator

    metadata = FunctionalFeatureMetadata(
        label=label,
        kind=kind,
        type_='array',
        axes=new_axes,
        order=tuple(new_order),
        extra_metadata={
            'ratio_axis': ratio_axis,
            'pairwise_mode': 'ordered',
        },
        kwargs=dict(label=label, ratio_axis=ratio_axis)
    )

    feature_structure = FunctionalFeatureStructure(
        values=new_values,
        metadata=metadata
    )

    # Update provenance
    feature_structure = update_provenance(input, feature_structure)


    return feature_structure


@functional_feature_decorator('functional_aggregate_feature', 'array')
def functional_aggregate_feature(input, label: Optional[str] = None, fun: Union[Callable, str] = np.mean, axisname: str ='epochs', max_numitem: Optional[int]=None) -> FunctionalFeatureStructure:
    """
    Aggregate the input data using a specified function along a specified axis.
    Parameters:
        input (FunctionalFeatureStructure): The input data to aggregate.
        label (Optional[str]): A unique name for this instance. Default is None.
        fun (Union[Callable, str]): The function to use for aggregation. Can be a callable or a string representing a numpy function.
        axisname (str): The name of the axis to aggregate along. Default is 'epochs'.
        max_numitem (Optional[int]): Maximum number of items to keep in the specified axis. Default is None.
    Returns:
        FeatureOutput: Contains aggregated data and metadata.
    """

    if isinstance(fun,str):
        # accept 'np.mean' or 'mean' (assumes it's a numpy function)
        try:
            fun=eval('np.'+fun)
        except:
            fun=eval('np.'+fun.replace('np.',''))
    if not callable(fun):
        raise ValueError(f"Function {fun} is not callable nor a string.")

    kind = get_kind_from_snake(currentframe().f_code.co_name)
    fun_name = fun.__name__ if hasattr(fun, '__name__') and fun.__name__ != '<lambda>' else 'anonymous'
    kind = input.metadata.kind + kind +snake_to_camel(axisname) + snake_to_camel(fun_name) + (str(max_numitem) if max_numitem is not None else '')

    output = deepcopy(input)
    output.metadata.kind = kind
    # input is the dict from np.load
    # a function like this could help for rois
    # spaces gets mapped to rois for example, and you modify the metadata appropiately
    axis = output.metadata.order
    axis = axis.index(axisname)

    if max_numitem is not None:
        if output['values'].shape[axis] >= max_numitem:
            # if we have more than max_numitem, we take the first max_numitem
            #TODO: also in theory we could give indexing array, then if we pass a single int use range, otherwise use the array of indices
            output['values'] = np.take(output['values'], indices=range(max_numitem), axis=axis)
        else:
            print(f"Warning: {output['values'].shape[axis]} items in axis {axisname} are less than max_numitem {max_numitem}.")

    # handle metadata appropriately
    output.values = fun(output.values,axis=axis)
    order = list(output.metadata.order)
    order.remove(axisname)
    output.metadata.order = tuple(order)
    del output.metadata.axes[axisname]


    output.metadata.kind = kind

    output = update_provenance(input, output)

    return output

@functional_feature_decorator('functional_binarize_along_axis_feature', 'array')
def functional_binarizer_feature(
    input: FunctionalFeatureStructure,
    *,
    label: Optional[str] = None,
    axisname: str = "times",
    threshold_fun: Union[Callable, str] = "median",
) -> FunctionalFeatureStructure:
    """
    Binarize the input data along a specified axis using a threshold function.

    Parameters:
        input (FunctionalFeatureStructure): The input data to binarize.
        label (Optional[str]): Optional label.
        axisname (str): Axis name to apply threshold along (default: "times").
        threshold_fun (Union[Callable, str]): Function or string for threshold (default: 'median').

    Returns:
        FunctionalFeatureStructure: Binarized output (0/1), with same axes.

    Notes:
    -----
    Axisname/Shape effects:

    +--------------------------+------------+----------+-------------------------------+-----------------------+
    | Data shape               | axisname   | axis_idx | Binarize “across”             | Threshold shape       |
    +==========================+============+==========+===============================+=======================+
    | (epochs, spaces, times)  | "epochs"   |    0     | all epochs at (space, time)   | (1, n_spaces, n_times)|
    +--------------------------+------------+----------+-------------------------------+-----------------------+
    | (epochs, spaces, times)  | "spaces"   |    1     | all spaces at (epoch, time)   | (n_epochs, 1, n_times)|
    +--------------------------+------------+----------+-------------------------------+-----------------------+
    | (epochs, spaces, times)  | "times"    |    2     | all times at (epoch, space)   | (n_epochs, n_spaces, 1)|
    +--------------------------+------------+----------+-------------------------------+-----------------------+
    | (spaces, times)          | "spaces"   |    0     | all spaces at (time)          | (1, n_times)          |
    +--------------------------+------------+----------+-------------------------------+-----------------------+
    | (spaces, times)          | "times"    |    1     | all times at (space)          | (n_spaces, 1)         |
    +--------------------------+------------+----------+-------------------------------+-----------------------+

    """
    # Get callable from string if needed
    if isinstance(threshold_fun, str):
        # Accept 'np.median', 'median', etc.
        try:
            threshold_fun = eval('np.' + threshold_fun)
        except Exception:
            threshold_fun = eval(threshold_fun)

    if not callable(threshold_fun):
        raise ValueError(f"threshold_fun {threshold_fun} is not callable or a string.")

    # Get axis index
    axis_idx = input.metadata.order.index(axisname)

    # Compute threshold along axis
    threshold_vals = threshold_fun(input.values, axis=axis_idx, keepdims=True)
    # Binarize: 1 if > threshold, 0 else
    binary = (input.values > threshold_vals).astype(int)

    # Copy and update metadata (keep axes and order the same)
    output = deepcopy(input)
    output.values = binary

    # Update kind to reflect binarization
    kind = get_kind_from_snake(currentframe().f_code.co_name)
    threshold_name = threshold_fun.__name__ if hasattr(threshold_fun, "__name__") else str(threshold_fun)
    output.metadata.kind = (
        (input.metadata.kind or "") + kind + snake_to_camel(axisname) + snake_to_camel(threshold_name)
    )
    output.metadata.label = label or output.metadata.label

    # Optionally update provenance
    output = update_provenance(input, output)

    return output



@functional_feature_decorator('functional_elementwise_feature', 'array')
def functional_elementwise_feature(
    input: FunctionalFeatureStructure,
    *,
    label: Optional[str] = None,
    fun: Union[Callable, str] = None,
    newkind: Optional[str] = None,
    newtype: Optional[str] = None
) -> FunctionalFeatureStructure:
    """
    Apply a function elementwise to the values array and update the type/kind if needed.

    Parameters
    ----------
    input : FunctionalFeatureStructure
        The input feature structure.
    label : str, optional
        Optional label.
    fun : callable or str
        The function to apply. If string, will be evaluated.
    newkind : str, optional
        If given, replaces the 'kind' field in metadata.
        If not given, the kind will be updated based on the function name.
    newtype : str, optional
        If given, replaces the 'type_' field in metadata.

    Returns
    -------
    FunctionalFeatureStructure
        New structure with function applied elementwise and (optionally) new type/kind.
    """
    if fun is None:
        raise ValueError("A function must be provided.")

    # Evaluate string function if needed
    if isinstance(fun, str):
        if fun.startswith("eval%"):
            fun = eval(fun.replace("eval%", ""))
        else:
            fun = eval(fun)  # Use cautiously!

    # Vectorize for elementwise operation
    vec_fun = np.vectorize(fun)

    # Copy input and apply function
    output = deepcopy(input)
    output.values = vec_fun(output.values)

    # Update kind
    if newkind is not None:
        output.metadata.kind = newkind
    else:
        kind = get_kind_from_snake(currentframe().f_code.co_name)
        fun_name = fun.__name__ if hasattr(fun, "__name__") else str(fun)
        output.metadata.kind = (
            (input.metadata.kind or "") + kind + snake_to_camel(fun_name)
        )

    # Update type if requested
    if newtype is not None:
        output.metadata.type_ = newtype

    output.metadata.label = label or output.metadata.label

    # Update provenance
    output = update_provenance(input, output)

    return output
