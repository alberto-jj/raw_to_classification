from .registry import ChainFeatureRegistry, FunctionalFeatureRegistry
from .base_chain import ChainFeatureStructure
from .base_functional import FunctionalFeatureStructure
from .base_functional import functional_load, functional_save
from .base_chain import process_chain_feature

#TODO: add mne basic processing feature used in chains (not saved): filter, resample, norm, etc.
# with that we could get rid of prefilter key in the config
# simiilarly we can use the same idea for prefoo in lempelziv, it would be a vectorize-like function

__all__ = [
    'ChainFeatureRegistry',
    'FunctionalFeatureRegistry',
    'ChainFeatureStructure',
    'FunctionalFeatureStructure',
    'functional_load',
    'functional_save',
    'process_chain_feature'
]