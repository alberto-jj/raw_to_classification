from typing import Dict, Type, List, Callable, Any
from .baseFeature import FeatureStructure
from .basePrimitive import PrimitiveFeatureStructure

class PrimitiveFeatureRegistry:
    """
    Holds named PrimitiveFeatureStructure (callable features).
    Unlike FeatureRegistry, which is for chains.
    """
    _primitive_features: Dict[str, Callable] = {}
    _primitive_types: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, primitive_type: str, func: Callable):
        cls._primitive_features[name] = func
        cls._primitive_types[name] = primitive_type

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._primitive_features[name]
    
    @classmethod
    def get_type(cls, name: str) -> str:
        return cls._primitive_types[name]
    
    @classmethod
    def list(cls) -> List[str]:
        return list(cls._primitive_features.keys())


class FeatureRegistry:
    """
    Holds named FeatureStructure chains (composable, declarative pipelines).
    Unlike FeatureRegistry, which is for callable features.
    """
    _chains: Dict[str, FeatureStructure] = {}

    @classmethod
    def register(cls, name: str, chain: FeatureStructure):
        cls._chains[name] = chain

    @classmethod
    def get(cls, name: str) -> FeatureStructure:
        return cls._chains[name]

    @classmethod
    def list(cls):
        return list(cls._chains.keys())
