from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import json
import pickle

from mne.io import Raw
from mne import Epochs
from mne.io import read_raw
from mne import read_epochs

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
            'specparam'). Example: "SpecparamNoKnee". Labels are camel case formatted.

        kind (str):
            The name of the feature family or conceptual category this functional feature belongs to.
            Example values: "spectrum", "specparam", "complexity".

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
            source dependencies, or visualization-specific attributes. Example entries:
              - "rendered_as": "html"
              - "source_feature": "spectrum"
              - "plot_type": "channels"

        kwargs (Dict[str, Any]):
            The parameters used to compute the functional feature. These are retained for reproducibility and
            interpretability. Example: {"method": "multitaper", "adaptive": True}

    Notes
    -----
    For non-array functional features such as inspectors or summaries, populate metadata fields as follows:
    TODO: Add mne.report or other types as needed
        https://mne.tools/stable/auto_tutorials/intro/70_report.html#sphx-glr-auto-tutorials-intro-70-report-py
        Field           What to populate
        -------------   ----------------------------------------------------------
        label           A unique name for this instance (e.g., 'spectrum_plot_summary')
        kind            The category of the source feature (e.g., 'spectrum')
        type_            Set to 'html' for visual output, 'dict' for summaries, etc.
        axes            Include only if relevant to the structure of the representation; otherwise use {}
        order           Use () if output is not a structured array
        extra_metadata  Describe the output format and context (e.g., {'rendered_as': 'html'})
        kwargs          Include any parameters used to configure the inspector or summarizer
    """
    label: str
    kind: str
    type_: str
    axes: Dict[str, Any]
    order: Tuple[str, ...]
    extra_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)


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
    if output_format == 'fif':
        if isinstance(object_to_save, Raw) or isinstance(object_to_save, Epochs):
            object_to_save.save(outputfile, overwrite=True)
        else:
            raise ValueError(f"Unknown MNE object type: {type(object_to_save)}")
    if output_format == 'npy':
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
    if output_format == 'fif':
        try:
            return read_raw(outputfile, preload=True)
        except:
            return read_epochs(outputfile, preload=True)

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

