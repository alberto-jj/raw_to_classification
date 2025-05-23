from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass
class ChainFeatureStructure:
    label: str
    overwrite: bool
    _type: str
    chain: List[Union[Dict[str, Any], str]]
    """
    label: str
        A unique name for this instance. This is useful for distinguishing between multiple variants of the same feature.
        Example: "SpecparamNoKnee". Labels are camel case formatted.

    overwrite: bool
        If True, the output will overwrite any existing data. If False, the output will be saved with a new name.
    _type: str
        The _type of the feature. This is used to determine how the feature should saved. E.g. "array"
    chain: list
        A list of dictionaries or strings representing the chain of features or functions to be applied.

        If you use a feature, then the dict will be:
        {
            'feature': 'FeatureName',
        }

        If you use a function, then the dict will be:
        {
            'function': 'function_name',
            'args': {
                'arg1': value1,
                'arg2': value2,
                ...
            }
        }


        If you use a feature, the output will be saved.
        If you use a function, the output will not be saved unless it is the last item in the chain.
        The input of the chain is assumed to be MNE object (Raw or Epochs) unless the first item is directly a feature.

    Example:
    ThisFeature:
        overwrite: False
        chain:
            - feature: AnotherFeature
            - function: some_function
              args:
                arg1: value1
                arg2: value2
    """
