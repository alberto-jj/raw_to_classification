from typing import Any, Dict, Optional
from .base_functional import FunctionalFeatureRegistry, functional_feature_to_format
from .base_functional import functional_load, functional_save
from .base_chain import ChainFeatureStructure, ChainFeatureRegistry
from copy import deepcopy

import os

def run_feature(input: Any,
                feature_structure: ChainFeatureStructure,
                file_bidspath: Optional[str] = None,
                inspect_only: bool = False,
                chain_feature_registry: Optional[Dict[str, Any]] = ChainFeatureRegistry,
                functional_feature_registry: Optional[Dict[str, Any]] = FunctionalFeatureRegistry
                ) -> Any:
    """
    Run a feature extraction process based on a chain of features and functions."""
    featdict = feature_structure
    overwrite = featdict.overwrite
    if not inspect_only:
        try:
            output = input.copy()
        except:
            output = deepcopy(input)
    else:
        output = None
    inspect_only_output = {}

    for i_f,stage in enumerate(featdict.chain):
        input_data = output
        if 'feature' in stage.keys():
            # is a feature that is saved or should be saved
            suffix = stage['feature']
            that_feature = chain_feature_registry.get(suffix)
            output_format = functional_feature_to_format(that_feature.type_)

            inner_featdict = chain_feature_registry.get(suffix)

            if file_bidspath is not None:
                input_suffix = file_bidspath.split('_')[-1]
                outputfile = file_bidspath.replace('_'+ input_suffix,f'_{suffix}.{output_format}')
            else:
                outputfile = None

            if file_bidspath is None or (file_bidspath and (not os.path.isfile(outputfile) or inner_featdict.overwrite)):
                if inspect_only:
                    inspect_only_output[suffix] = False
                    continue
                print(f'Feature {suffix} not found, computing it...')
                output = run_feature(input_data,
                                        inner_featdict,
                                        file_bidspath,
                                        inspect_only=inspect_only,
                                        chain_feature_registry=chain_feature_registry,
                                        functional_feature_registry=functional_feature_registry)
                if outputfile is not None:
                    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                    functional_save(output, outputfile, output_format)
            else:
                if inspect_only:
                    inspect_only_output[suffix] = True
                    continue
                print(f'Already Exists:{outputfile}')
                output = functional_load(outputfile, output_format)
                inspect_only_output[suffix] = True

        if 'function' in stage.keys():
            inner_featdict = stage
            if i_f == len(featdict.chain)-1:
                # Last stage, assume we want to save it with the feature name
                suffix = featdict.label
                functional_type = functional_feature_registry.get_type(stage['function'])

                if file_bidspath is not None:
                    input_suffix = file_bidspath.split('_')[-1]
                    output_format = functional_feature_to_format(functional_type)
                    outputfile = file_bidspath.replace('_'+ input_suffix,f'_{suffix}.{output_format}')
                else:
                    outputfile = None
                if file_bidspath is None or (file_bidspath and (not os.path.isfile(outputfile) or overwrite)):
                    if inspect_only:
                        inspect_only_output[suffix] = False
                        continue
                    fun = functional_feature_registry.get(inner_featdict['function'])
                    if isinstance(fun,str):
                        fun=eval(fun)
                    output = fun(input_data,**inner_featdict['args'])

                    if outputfile is not None:
                        os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                        functional_save(output, outputfile, output_format)
                    
                else:
                    if inspect_only:
                        inspect_only_output[suffix] = True
                        continue
                    print(f'Already Exists:{outputfile}')
                    output = functional_load(outputfile, output_format)
            else:
                if inspect_only:
                    continue
                innerfun=eval(f"inner_featdict['function']")
                innerfun=eval(innerfun)
                output = innerfun(input_data,**inner_featdict['args'])
        input_data = output
    if inspect_only:
        output=inspect_only_output
    return output
