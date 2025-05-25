from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class ChainFeatureStructure:
    label: str
    overwrite: bool
    type_: str
    chain: List[Union[Dict[str, Any], str]]
    """
    label: str
        A unique name for this instance assigned by the user. This is useful for distinguishing between multiple variants of the same feature.
        Example: "SpecparamNoKnee". Labels are camel case formatted.

    overwrite: bool
        If True, the output will overwrite any existing data. If False, the output will be saved with a new name.
    type_: str
        The type_ of the feature. This is used to determine how the feature should saved. E.g. "array"
    chain: list
        A list of dictionaries or strings representing the chain of features or functions to be applied.

        If you use a feature, then the dict will be:
        {
            'feature': 'FeatureName', (FeatureName must be registered in the ChainFeatureRegistry)
        }

        If you use a function, then the dict will be:
        {
            'function': 'function_name', ( function_name must be registered in the FunctionalFeatureRegistry)
            'args': {
                'arg1': value1,
                'arg2': value2,
                ...
            }
        }


        If you use a feature, the output will be saved.
        If you use a function, the output will not be saved unless it is the last item in the chain.
        The input of the chain is assumed to be MNE object (BaseRaw or BaseEpochs) unless the first item is directly a feature.

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


class ChainFeatureRegistry:
    """
    Holds named ChainFeatureStructure chains (composable, declarative pipelines).
    Unlike FunctionalFeatureRegistry, which is for callable features.
    """
    _chains: Dict[str, ChainFeatureStructure] = {}

    @classmethod
    def register(cls, name: str, chain: ChainFeatureStructure):
        cls._chains[name] = chain

    @classmethod
    def get(cls, name: str) -> ChainFeatureStructure:
        return cls._chains[name]

    @classmethod
    def list(cls):
        return list(cls._chains.keys())


def chain_dict_to_chain_structure(chain_dict: Dict[str, Any]) -> ChainFeatureStructure:
    """Convert a chain dictionary to a chain structure.

    Parameters
    ----------
    chain_dict : dict
        The chain dictionary to convert.

    Returns
    -------
    ChainFeatureStructure
        The converted chain structure.
    """
    chain_structure = ChainFeatureStructure(
        label=chain_dict['label'],
        overwrite=chain_dict['overwrite'],
        type_=chain_dict['type_'],
        chain=chain_dict['chain']
    )
    return chain_structure
