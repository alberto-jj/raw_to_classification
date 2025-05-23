
from copy import deepcopy
from mne.io import Raw
from mne import Epochs
import numpy as np
from copy import deepcopy
from .registry import PrimitiveFeatureRegistry
from typing import Optional, Dict, Any, Union,Callable
from .basePrimitive import PrimitiveFeatureMetadata, PrimitiveFeatureStructure
from .utils import snake_to_camel

def primitive_aggregate_feature(input, label: Optional[str] = None, fun: Union[Callable, str] = np.mean, axisname: str ='epochs', max_numitem: Optional[int]=None) -> PrimitiveFeatureStructure:
    """
    Aggregate the input data using a specified function along a specified axis.
    Parameters:
        input (Union[Epochs, Raw]): The input data to compute the power spectrum from.
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

    output = deepcopy(input)
    # input is the dict from np.load
    # a function like this could help for rois
    # spaces gets mapped to rois for example, and you modify the metadata appropiately
    axis = output.metadata.order
    axis = axis.index(axisname)

    if max_numitem is not None:
        if output['values'].shape[axis] >= max_numitem:
            # if we have more than max_numitem, we take the first max_numitem
            output['values'] = np.take(output['values'], indices=range(max_numitem), axis=axis)
        else:
            print(f"Warning: {output['values'].shape[axis]} items in axis {axisname} are less than max_numitem {max_numitem}.")

    # handle metadata appropriately
    output.values = fun(output.values,axis=axis)
    order = list(output.metadata.order)
    order.remove(axisname)
    output.metadata.order = tuple(order)
    del output.metadata.axes[axisname]
    if label is not None:
        output.metadata.label = label
    elif isinstance(output.metadata.label,str):
        output.metadata.label = output.metadata.label + snake_to_camel(axisname) + snake_to_camel(fun.__name__) + str(max_numitem) if max_numitem is not None else ''
    return output

PrimitiveFeatureRegistry.register(primitive_aggregate_feature.__name__, 'array', primitive_aggregate_feature)