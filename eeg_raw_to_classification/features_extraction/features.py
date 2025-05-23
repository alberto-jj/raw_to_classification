from .baseFeature import FeatureStructure
from .registry import FeatureRegistry
import numpy as np
import os

# maybe we should separate into family chains (e.g. spectralChains.py, timeChains.py, etc)
# but then how to handle the imports without hardcoding them in chains.py?


SpectrumMultitaper = FeatureStructure(
    label='SpectrumMultitaper',
    overwrite=False,
    _type='array',
    chain=[
        dict(
            function='primitive_spectrum_feature',
            args=dict(
                method='multitaper',
                mne_kwargs=dict(
                    adaptive=False,
                    low_bias=True,
                    normalization='full',
                    verbose=0
                )
            )
        )
    ]
)

FeatureRegistry.register(SpectrumMultitaper.label, SpectrumMultitaper)

SpectrumMultitaperAverage = FeatureStructure(
    label='SpectrumMultitaperAverage',
    overwrite=False,
    _type='array',
    chain=[
        dict(feature='SpectrumMultitaper'),
        dict(
            function='primitive_aggregate_feature',
            args=dict(
                fun=np.mean,
                axisname='epochs',
                max_numitem=None
            )
        )
    ]
)

FeatureRegistry.register(SpectrumMultitaperAverage.label, SpectrumMultitaperAverage)