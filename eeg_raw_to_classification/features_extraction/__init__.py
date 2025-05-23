from .registry import PrimitiveFeatureRegistry, FeatureRegistry
from .basePrimitive import PrimitiveFeatureStructure, PrimitiveFeatureMetadata
from .baseFeature import FeatureStructure
from .utils import get_mne_metadata, snake_to_camel, primitive_feature_to_format, primitive_save, primitive_load
from typing import Optional, Dict, Any, Union, Callable
from .spectralPrimitive import *
from .aggregatePrimitive import *
from .features import *



def process_feature(input: Any,
                    feature_structure: FeatureStructure,
                    relevantpath: Optional[str] = None,
                    inspect_only: bool = False,
                    feature_registry: Optional[Dict[str, Any]] = FeatureRegistry,
                    primitive_feature_registry: Optional[Dict[str, Any]] = PrimitiveFeatureRegistry
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
            that_feature = feature_registry.get(suffix)
            output_format = primitive_feature_to_format(that_feature._type)

            inner_featdict = feature_registry.get(suffix)

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
                output = process_feature(input_data,
                                        inner_featdict,
                                        relevantpath,
                                        inspect_only=inspect_only,
                                        feature_registry=feature_registry,
                                        primitive_feature_registry=primitive_feature_registry)
                if outputfile is not None:
                    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                    primitive_save(output, outputfile, output_format)
            else:
                if inspect_only:
                    inspect_only_output.append(True)
                    continue
                print(f'Already Exists:{outputfile}')
                output = primitive_load(outputfile, output_format)
                inspect_only_output.append(True)

        if 'function' in stage.keys():
            inner_featdict = stage
            if i_f == len(featdict.chain)-1:
                # Last stage, assume we want to save it with the feature name
                suffix = featdict.label
                primitive_type = primitive_feature_registry.get_type(stage['function'])

                if relevantpath is not None:
                    input_suffix = relevantpath.split('_')[-1]
                    output_format = primitive_feature_to_format(primitive_type)
                    outputfile = relevantpath.replace('_'+ input_suffix,f'_{suffix}.{output_format}')
                else:
                    outputfile = None
                if relevantpath is None or (relevantpath and (not os.path.isfile(outputfile) or overwrite)):
                    if inspect_only:
                        inspect_only_output.append(False)
                        continue
                    fun = primitive_feature_registry.get(inner_featdict['function'])
                    if isinstance(fun,str):
                        fun=eval(fun)
                    output = fun(input_data,**inner_featdict['args'])

                    if outputfile is not None:
                        os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                        primitive_save(output, outputfile, output_format)
                    
                else:
                    if inspect_only:
                        inspect_only_output.append(True)
                        continue
                    print(f'Already Exists:{outputfile}')
                    output = primitive_load(outputfile, output_format)
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
