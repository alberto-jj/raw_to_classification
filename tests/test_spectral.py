import pytest
import mne
import numpy as np

import eeg_raw_to_classification.feature_extraction.spectral_functionals as pf

# Optional: If decorators register to a central registry, this helps test listing
import eeg_raw_to_classification.feature_extraction as fe


@pytest.fixture(scope="module")
def raw_epochs():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False).crop(tmax=5)

    events = mne.find_events(raw, stim_channel="STI 014")
    raw = raw.pick_types(eeg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, tmin=-0.5, tmax=0.5, preload=True, verbose=False)

    # Standardize channel names using standard_1020 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    name_map = {ch: montage.ch_names[i] for i, ch in enumerate(raw.ch_names)}
    raw.rename_channels(name_map)
    epochs.rename_channels(name_map)

    num_channels = 3
    epochs = epochs.copy().pick_channels(epochs.ch_names[:num_channels])

    return raw, epochs


def test_spectrum_feature_inspect(raw_epochs):
    _, epochs = raw_epochs
    result = pf.functional_spectrum_feature(epochs, method="multitaper", mne_kwargs={'adaptive': False})
    result.inspect()


def test_bandspectrum_feature_inspect(raw_epochs):
    _, epochs = raw_epochs
    spectrum = pf.functional_spectrum_feature(epochs, method="multitaper", mne_kwargs={'adaptive': False})
    bands = {
        'delta': [0.5, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 45]
    }
    band_result = pf.functional_bandspectrum_feature(spectrum, bands=bands, relative=False)
    band_result.inspect()


def test_fooof_feature_inspect(raw_epochs):
    _, epochs = raw_epochs
    spectrum = pf.functional_spectrum_feature(epochs, method="multitaper", mne_kwargs={'adaptive': False})
    fooof_result = pf.functional_fooof_feature(spectrum, internal_kwargs={'FOOOF': {}, 'fit': {}})
    fooof_result.inspect()


@pytest.mark.parametrize("component", ["original", "aperiodic", "oscillatory"])
def test_fooof_component_inspect(raw_epochs, component):
    _, epochs = raw_epochs
    spectrum = pf.functional_spectrum_feature(epochs, method="multitaper", mne_kwargs={'adaptive': False})
    fooof_struct = pf.functional_fooof_feature(spectrum, internal_kwargs={'FOOOF': {}, 'fit': {}})
    comp_result = pf.functional_fooof_component_feature(fooof_struct, component=component)
    comp_result.inspect()


if __name__ == "__main__":
    import sys
    import pytest


    for i in fe.FunctionalFeatureRegistry.list():
        print(i)

    for i in fe.ChainFeatureRegistry.list():
        print(i)


    # Run only this file
    sys.exit(pytest.main([__file__]))
