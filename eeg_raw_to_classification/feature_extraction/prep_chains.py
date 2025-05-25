from .base_chain import ChainFeatureStructure, ChainFeatureRegistry, chain_dict_to_chain_structure
import numpy as np
import yaml
this_file = __file__
this_yaml = this_file.replace('.py', '.yml')

yaml_definitions = yaml.safe_load(open(this_yaml, 'r'))
# Register the chain features

for name, chain in yaml_definitions.items():
    # Convert the chain to a ChainFeatureStructure
    # Register the chain
    chain_structure = chain_dict_to_chain_structure(chain)
    ChainFeatureRegistry.register(name, chain_structure)