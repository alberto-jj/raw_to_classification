from .base_functional import FunctionalFeatureRegistry
from .base_chain import ChainFeatureRegistry
from .base_chain import ChainFeatureStructure

def functional_feature(name: str, functional_type: str):
    def decorator(func):
        FunctionalFeatureRegistry.register(name, functional_type, func)
        return func
    return decorator

def chain_feature(name: str):
    def decorator(chain_obj: ChainFeatureStructure):
        ChainFeatureRegistry.register(name, chain_obj)
        return chain_obj
    return decorator
