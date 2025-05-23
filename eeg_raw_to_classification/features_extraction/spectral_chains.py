from .base_chain import ChainFeatureStructure
from .registry import ChainFeatureRegistry
import numpy as np
import os

# maybe we should separate into family chains (e.g. spectralChains.py, timeChains.py, etc)
# but then how to handle the imports without hardcoding them in chains.py?


SpectrumMultitaper = ChainFeatureStructure(
    label='SpectrumMultitaper',
    overwrite=False,
    type_='array',
    chain=[
        dict(
            function='functional_spectrum_feature',
            args=dict(
                method='multitaper',
                label='SpectrumMultitaper',
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

ChainFeatureRegistry.register(SpectrumMultitaper.label, SpectrumMultitaper)

SpectrumMultitaperAverage = ChainFeatureStructure(
    label='SpectrumMultitaperAverage',
    overwrite=False,
    type_='array',
    chain=[
        dict(feature='SpectrumMultitaper'),
        dict(function='functional_aggregate_feature',
            args=dict(
                label='SpectrumMultitaperAverage',
                fun=np.mean,
                axisname='epochs',
                max_numitem=None
            )
        )
    ]
)

ChainFeatureRegistry.register(SpectrumMultitaperAverage.label, SpectrumMultitaperAverage)