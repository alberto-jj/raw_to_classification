from typing import Dict, Type, List, Callable, Any
from .baseChain import ChainFeatureStructure
from .baseFunctional import FunctionalFeatureStructure

class FunctionalFeatureRegistry:
    """
    Holds named FunctionalFeatureStructure (callable features).
    Unlike ChainFeatureRegistry, which is for chains.
    """
    _functional_features: Dict[str, Callable] = {}
    _functional_types: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, functional_type: str, func: Callable):
        cls._functional_features[name] = func
        cls._functional_types[name] = functional_type

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._functional_features[name]
    
    @classmethod
    def get_type(cls, name: str) -> str:
        return cls._functional_types[name]
    
    @classmethod
    def list(cls) -> List[str]:
        return list(cls._functional_features.keys())


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
