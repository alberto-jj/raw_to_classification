from .base_chain import ChainFeatureStructure, ChainFeatureRegistry
from .base_functional import FunctionalFeatureStructure, FunctionalFeatureRegistry
from .base_functional import functional_load, functional_save
from .base_functional import functional_feature_to_format
from .run import run_feature

__all__ = [
    'ChainFeatureRegistry',
    'FunctionalFeatureRegistry',
    'ChainFeatureStructure',
    'FunctionalFeatureStructure',
    'functional_load',
    'functional_save',
    'functional_feature_to_format',
    'run_feature',
]

# Dynamically import all submodules to trigger decorator-based registration
import pkgutil
import importlib

def import_submodules(package_name):
    """Import all submodules in a package to trigger decorators and side effects."""
    package = importlib.import_module(package_name)
    for _, modname, _ in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
        importlib.import_module(modname)

# Automatically load all submodules under this package
import_submodules(__name__)

#TODO: add mne basic processing feature used in chains (not saved): filter, resample, norm, etc.
# with that we could get rid of prefilter key in the config
# simiilarly we can use the same idea for prefoo in lempelziv, it would be a vectorize-like function

# Add all registered functional and chain features to namespace and __all__
for name in FunctionalFeatureRegistry.list():
    globals()[name] = FunctionalFeatureRegistry.get(name)
    __all__.append(name)

for name in ChainFeatureRegistry.list():
    globals()[name] = ChainFeatureRegistry.get(name)
    __all__.append(name)