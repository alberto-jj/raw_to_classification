from .base_chain import ChainFeatureStructure, ChainFeatureRegistry
import numpy as np
import yaml
this_file = __file__
this_yaml = this_file.replace('.py', '.yml')

yaml_definitions = yaml.safe_load(open(this_yaml, 'r'))
# Register the chain features

for name, chain in yaml_definitions.items():
    # Convert the chain to a ChainFeatureStructure
    chain_structure = ChainFeatureStructure(
        label=chain['label'],
        overwrite=chain['overwrite'],
        type_=chain['type_'],
        chain=chain['chain']
    )
    # Register the chain
    ChainFeatureRegistry.register(name, chain_structure)