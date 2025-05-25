from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import json
import pickle

from mne.io import BaseRaw
from mne import Epochs
from mne.io import read_raw
from mne import read_epochs
import os
# ---------------------------
# Data Structures
# ---------------------------

@dataclass
class FunctionalFeatureMetadata:
    """
    Describes the structure, configuration, and context of a functional feature's output.

    This metadata object applies to all functional features, including those that return numerical arrays
    (e.g., power spectra), visualizations (e.g., HTML plots), or summary representations (e.g., dictionaries).
    It ensures consistency, traceability, and appropriate handling of functional feature results across the pipeline.

    Attributes:
        label (str):
            A user-defined identifier for the specific functional feature instance. This is useful for distinguishing
            between multiple variants of the same functional feature family (e.g., different parameterizations of
            'Specparam'). Example: "SpecparamNoKnee". Labels are camel case formatted.

        kind (str):
            The name of the feature family or conceptual category this functional feature belongs to. Handled in the code.
            Example values: "Spectrum", "Specparam", "Complexity".
            Should be auto populated from the function name using utils.get_kind_from_snake(inspect.currentframe().f_code.co_name)

        type_ (str):
            A descriptor of the output format. This informs how the `values` should be interpreted and handled.
            Common values include:
              - "array" → for standard `np.ndarray` outputs
              - "html" → for visual or inspector outputs
              - "dict" → for JSON-serializable summary outputs
              - TODO: Add mne.report or other types as needed

        axes (Dict[str, Any]):
            A mapping from named axes (e.g., "epochs", "channels", "frequencies") to their associated labels or values.
            For array-based functional features, this reflects the structure of the output tensor.
            For non-array functional features, this may be empty or omitted.

        order (Tuple[str, ...]):
            Specifies the order of dimensions in the output `values`. Should correspond to keys in `axes`.
            For non-array outputs, use an empty tuple.

        extra_metadata (Optional[Dict[str, Any]]):
            Additional contextual information about the functional feature output. This may include rendering hints,
            source dependencies, or visualization-specific attributes.

        kwargs (Dict[str, Any]):
            The parameters used to compute the functional feature. These are retained for reproducibility and
            interpretability. Example: {"method": "multitaper", "adaptive": True}

        provenance (Optional[List[Any]]):
            A list of provenance information, which may include references to the original data or processing steps.
            If a functional feature is derived from another, append the input metadata (without its provenance) here as the last item.


    Notes
    -----
    For non-array functional features such as inspectors or summaries, populate metadata fields as follows:
    TODO: Add mne.report or other types as needed
        https://mne.tools/stable/auto_tutorials/intro/70_report.html#sphx-glr-auto-tutorials-intro-70-report-py
        Field           What to populate
        -------------   ----------------------------------------------------------
        label           A unique name for this instance (e.g., 'spectrum_plot_summary')
        kind            The category of the source feature. Example: 'Spectrum'.
        type_            Set to 'html' for visual output, 'dict' for summaries, etc.
        axes            Include only if relevant to the structure of the representation; otherwise use {}
        order           Use () if output is not a structured array
        extra_metadata  Describe the output format and context (e.g., {'rendered_as': 'html'})
        kwargs          Include any parameters used to configure the inspector or summarizer
        provenance      Include the input metadata (without its provenance) as the last item
        """
    label: str
    kind: str
    type_: str
    axes: Dict[str, Any]
    order: Tuple[str, ...]
    extra_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[List[Any]] = field(default_factory=list)


@dataclass
class FunctionalFeatureStructure:
    """
    A standardized container returned by any functional feature in the system, including transformations,
    aggregators, and inspectors.

    This class wraps the actual computed output along with rich metadata that describes
    the structure, context, and parameters of the computation.

    Attributes:
        values (Union[np.ndarray, str, Dict, Any]):
            The main result of the functional feature computation. Its type_ depends on the nature of the functional feature:
            
            - np.ndarray: typical for core functional features (e.g., spectral power, entropy)
            - str: for visual or HTML-based inspectors
            - Dict: for summary statistics or JSON-serializable reports
            - Any: allows future extension (e.g., plots, figures, file paths)

        metadata (FunctionalFeatureMetadata):
            Metadata describing the axes, dimensions, and parameters associated with the computation.
            For inspector features or non-array outputs, `axes` and `order` can be empty, but `type_`,
            `kwargs`, and `extra_metadata` should still describe the context of the result.
    """
    values: Union[np.ndarray, str, Dict, Any]
    metadata: FunctionalFeatureMetadata

    def inspect(self):
        shape = None
        try:
            shape = self.values.shape
        except AttributeError:
            try:
                shape = self.values._data.shape
            except AttributeError:
                pass
        print('values shape:', shape)
        print('kind:', self.metadata.kind)
        print('label:', self.metadata.label)
        print('order:', self.metadata.order)
        print('axes:', list(self.metadata.axes.keys()))
        print('provenance:', self.metadata.provenance)

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


def inspect_example(input: Optional[FunctionalFeatureStructure] = None) -> str:
    """
    Generate an HTML string with a visual summary of the feature output.
    Should be overridden in subclasses for actual visualization.

    Parameters:
        output (ChainFeatureStructure): The output to inspect. Defaults to the last computed one.

    Returns:
        str: HTML-formatted inspection report.
    """
    return f"<p><b>{input.metadata.type_}</b>: no custom inspect defined.</p>"



def functional_feature_to_format(this_type:str):
    """Convert a feature type_ to a standardized format.

    Parameters
    ----------
    this_type : str
        The input feature type_.

    Returns
    -------
    str
        The converted feature type_ in a standardized format.
    """
    if this_type == 'array':
        return 'npy'
    elif this_type == 'html':
        return 'html'
    elif this_type == 'dict':
        return 'json'
    elif this_type == 'pickle':
        return 'pickle'
    elif this_type == 'mne':
        return 'mne'
    else:
        raise ValueError(f"Unknown feature type_: {this_type}")

def functional_save(object_to_save, outputfile, output_format):
    """Save the object_to_save in the specified format.

    Parameters
    ----------
    object_to_save : any
        The output to save.
    outputfile : str
        The filename to save the output to.
    output_format : str
        The format to save the output in.
    """
    if output_format == 'mne':
        mne_object = object_to_save.values
        metadata = object_to_save.metadata
        basename = os.path.basename(outputfile)
        name = os.path.splitext(basename)[0]
        extension = os.path.splitext(basename)[1]
        if extension != '.fif':
            outputfile = os.path.join(os.path.dirname(outputfile), name + '.fif')
        mne_object.save(outputfile, overwrite=True)
        object_to_save.values = outputfile
        full_path = os.path.join(os.path.dirname(outputfile), name + '.mne')
        with open(full_path, 'wb') as f:
            pickle.dump(object_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif output_format == 'npy':
        np.save(outputfile,object_to_save,allow_pickle=True)
    elif output_format == 'json':
        with open(outputfile, 'w') as f:
            json.dump(object_to_save, f, indent=4)
    elif output_format == 'html':
        with open(outputfile, 'w') as f:
            f.write(object_to_save)
    elif output_format == 'pickle':
        with open(outputfile, 'wb') as f:
            pickle.dump(object_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unknown format: {output_format}")

def functional_load(outputfile, output_format):
    """Load the output from the specified format.

    Parameters
    ----------
    outputfile : str
        The filename to load the output from.
    output_format : str
        The format to load the output in.

    Returns
    -------
    any
        The loaded output.
    """
    if output_format == 'mne':
        if not os.path.exists(outputfile):
            raise ValueError(f"File not found: {outputfile}")
        
        basedir = os.path.dirname(outputfile)
        basename = os.path.basename(outputfile)
        name = os.path.splitext(basename)[0]
        extension = os.path.splitext(basename)[1]
        if extension == '.fif':
            fif_file = os.path.join(basedir, name + '.fif')
            meta_file = os.path.join(basedir, name + '.mne')
        elif extension == '.mne':
            meta_file = os.path.join(basedir, name + '.mne')
            fif_file = os.path.join(basedir, name + '.fif')
        else:
            raise ValueError(f"Failed to load MNE object from {outputfile}. File must end with .fif or .mne")

        try:
            mne_object = read_raw(fif_file, preload=True)
        except:
            print(f"Could not load raw from {fif_file}. Trying to load epochs.")
            mne_object = read_epochs(fif_file, preload=True)
        with open(meta_file, 'rb') as f:
            object_to_load = pickle.load(f)
        object_to_load.values = mne_object
        return object_to_load
    if output_format == 'npy':
        return np.load(outputfile,allow_pickle=True).item()
    elif output_format == 'json':
        with open(outputfile, 'r') as f:
            return json.load(f)
    elif output_format == 'html':
        with open(outputfile, 'r') as f:
            return f.read()
    elif output_format == 'pickle':
        with open(outputfile, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {output_format}")

