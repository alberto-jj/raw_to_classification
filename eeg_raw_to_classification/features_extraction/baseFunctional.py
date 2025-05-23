from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import os
#import pandas as pd

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

        _type (str):
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
        _type            Set to 'html' for visual output, 'dict' for summaries, etc.
        axes            Include only if relevant to the structure of the representation; otherwise use {}
        order           Use () if output is not a structured array
        extra_metadata  Describe the output format and context (e.g., {'rendered_as': 'html'})
        kwargs          Include any parameters used to configure the inspector or summarizer
    """
    label: str
    kind: str
    _type: str
    axes: Dict[str, Any]
    order: Tuple[str, ...]
    extra_metadata: Optional[Dict[str, Any]] = None
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
            The main result of the functional feature computation. Its _type depends on the nature of the functional feature:
            
            - np.ndarray: typical for core functional features (e.g., spectral power, entropy)
            - str: for visual or HTML-based inspectors
            - Dict: for summary statistics or JSON-serializable reports
            - Any: allows future extension (e.g., plots, figures, file paths)

        metadata (FunctionalFeatureMetadata):
            Metadata describing the axes, dimensions, and parameters associated with the computation.
            For inspector features or non-array outputs, `axes` and `order` can be empty, but `_type`,
            `kwargs`, and `extra_metadata` should still describe the context of the result.
    """
    values: Union[np.ndarray, str, Dict, Any]
    metadata: FunctionalFeatureMetadata


def inspect_example(input: Optional[FunctionalFeatureStructure] = None) -> str:
    """
    Generate an HTML string with a visual summary of the feature output.
    Should be overridden in subclasses for actual visualization.

    Parameters:
        output (ChainFeatureStructure): The output to inspect. Defaults to the last computed one.

    Returns:
        str: HTML-formatted inspection report.
    """
    return f"<p><b>{input.metadata._type}</b>: no custom inspect defined.</p>"

