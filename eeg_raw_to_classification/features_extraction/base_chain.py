from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import os
from .base_chain import ChainFeatureStructure
from .registry import ChainFeatureRegistry, FunctionalFeatureRegistry
from .utils import functional_feature_to_format, functional_save, functional_load

@dataclass
class ChainFeatureStructure:
    label: str
    overwrite: bool
    type_: str
    chain: List[Union[Dict[str, Any], str]]
    """
    label: str
        A unique name for this instance. This is useful for distinguishing between multiple variants of the same feature.
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


def process_chain_feature(input: Any,
                    feature_structure: ChainFeatureStructure,
                    relevantpath: Optional[str] = None,
                    inspect_only: bool = False,
                    chain_feature_registry: Optional[Dict[str, Any]] = ChainFeatureRegistry,
                    functional_feature_registry: Optional[Dict[str, Any]] = FunctionalFeatureRegistry
                    ) -> Any:
    featdict = feature_structure
    overwrite = featdict.overwrite
    if not inspect_only:
        try:
            output = input.copy()
        except:
            output = deepcopy(input)
    else:
        output = None
    inspect_only_output = []

    for i_f,stage in enumerate(featdict.chain):
        input_data = output
        if 'feature' in stage.keys():
            # is a feature that is saved or should be saved
            suffix = stage['feature']
            that_feature = chain_feature_registry.get(suffix)
            output_format = functional_feature_to_format(that_feature.type_)

            inner_featdict = chain_feature_registry.get(suffix)

            if relevantpath is not None:
                input_suffix = relevantpath.split('_')[-1]
                outputfile = relevantpath.replace('_'+ input_suffix,f'_{suffix}.{output_format}')
            else:
                outputfile = None

            if relevantpath is None or (relevantpath and (not os.path.isfile(outputfile) or inner_featdict.overwrite)):
                if inspect_only:
                    inspect_only_output.append(False)
                    continue
                print(f'Feature {suffix} not found, computing it...')
                output = process_chain_feature(input_data,
                                        inner_featdict,
                                        relevantpath,
                                        inspect_only=inspect_only,
                                        chain_feature_registry=chain_feature_registry,
                                        functional_feature_registry=functional_feature_registry)
                if outputfile is not None:
                    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                    functional_save(output, outputfile, output_format)
            else:
                if inspect_only:
                    inspect_only_output.append(True)
                    continue
                print(f'Already Exists:{outputfile}')
                output = functional_load(outputfile, output_format)
                inspect_only_output.append(True)

        if 'function' in stage.keys():
            inner_featdict = stage
            if i_f == len(featdict.chain)-1:
                # Last stage, assume we want to save it with the feature name
                suffix = featdict.label
                functional_type = functional_feature_registry.get_type(stage['function'])

                if relevantpath is not None:
                    input_suffix = relevantpath.split('_')[-1]
                    output_format = functional_feature_to_format(functional_type)
                    outputfile = relevantpath.replace('_'+ input_suffix,f'_{suffix}.{output_format}')
                else:
                    outputfile = None
                if relevantpath is None or (relevantpath and (not os.path.isfile(outputfile) or overwrite)):
                    if inspect_only:
                        inspect_only_output.append(False)
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
                        inspect_only_output.append(True)
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
