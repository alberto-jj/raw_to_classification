import eeg_raw_to_classification.feature_extraction as fe
import eeg_raw_to_classification.feature_extraction.prep_functionals as pf
import mne
import os
import pytest

@pytest.fixture(scope="module")
def raw_epochs():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False, preload=True).crop(tmax=60)

    events = mne.find_events(raw, stim_channel="STI 014")
    raw = raw.pick_types(eeg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, tmin=-0.5, tmax=0.5, preload=True, verbose=False)

    # Reassign channel names using standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    name_map = {ch: montage.ch_names[i] for i, ch in enumerate(raw.ch_names)}
    raw.rename_channels(name_map)
    epochs.rename_channels(name_map)

    return raw, epochs


def test_cast_to_structure(raw_epochs):
    raw, epochs = raw_epochs
    pf.functional_cast_to_structure_feature(raw).inspect()
    pf.functional_cast_to_structure_feature(epochs).inspect()

def test_channel_zscore(raw_epochs):
    raw, epochs = raw_epochs
    pf.functional_channel_zscore_feature(raw).inspect()
    pf.functional_channel_zscore_feature(epochs).inspect()

def test_filter_feature(raw_epochs):
    raw, epochs = raw_epochs
    pf.functional_filter_feature(raw, mne_kwargs={'l_freq': 1, 'h_freq': 40}).inspect()
    pf.functional_filter_feature(epochs, mne_kwargs={'l_freq': 1, 'h_freq': 40}).inspect()

def test_make_fixed_length_epochs(raw_epochs):
    raw, _ = raw_epochs
    pf.functional_make_fixed_length_epochs_feature(raw, mne_kwargs={'duration': 1.0}).inspect()

def test_notch_filter_on_raw(raw_epochs):
    raw, epochs = raw_epochs
    pf.functional_notch_filter_feature(raw, mne_kwargs={'freqs': [50, 100]}).inspect()

    with pytest.raises(ValueError):
        pf.functional_notch_filter_feature(epochs, mne_kwargs={'freqs': [50, 100]}).inspect()

def test_notch_filter_error_on_epochs(raw_epochs):
    _, epochs = raw_epochs
    with pytest.raises(ValueError):
        pf.functional_notch_filter_feature(epochs, mne_kwargs={'freqs': [50, 100]}).inspect()

def test_resample(raw_epochs):
    raw, epochs = raw_epochs
    pf.functional_resample_feature(raw, mne_kwargs={'sfreq': 512}).inspect()
    pf.functional_resample_feature(epochs, mne_kwargs={'sfreq': 512}).inspect()

def test_set_montage(raw_epochs):
    raw, epochs = raw_epochs
    pf.functional_set_montage_feature(raw, mne_kwargs={'montage_kind': 'standard_1020'}).inspect()
    pf.functional_set_montage_feature(epochs, mne_kwargs={'montage_kind': 'standard_1020'}).inspect()

def test_standardize_channel_names(raw_epochs):
    raw, epochs = raw_epochs
    pf.functional_standardize_channel_names_feature(raw).inspect()
    pf.functional_standardize_channel_names_feature(epochs).inspect()



if __name__ == "__main__":
    import sys
    import pytest


    for i in fe.FunctionalFeatureRegistry.list():
        print(i)

    for i in fe.ChainFeatureRegistry.list():
        print(i)


    # Run only this file
    sys.exit(pytest.main([__file__]))
