import pytest
import mne
import numpy as np

# (Optional) Import the registry if you want to test feature listing
import eeg_raw_to_classification.feature_extraction as fe

features = [i for i in fe.FunctionalFeatureRegistry.list() if i.startswith("functional_") and i.endswith("_feature") and ('neurokit2' in i)] # or 'antropy' in i
@pytest.fixture(scope="module")
def raw_epochs():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False).crop(tmax=5)

    events = mne.find_events(raw, stim_channel="STI 014")
    raw = raw.pick_types(eeg=True, exclude="bads")
    epochs = mne.Epochs(raw, events, tmin=-0.5, tmax=0.5, preload=True, verbose=False)
    raw = raw.resample(50, npad="auto")  # Resample to 100 Hz for faster processing
    epochs = epochs.resample(50, npad="auto")  # Resample epochs to 100 Hz

    # Standardize channel names using standard_1020 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    name_map = {ch: montage.ch_names[i] for i, ch in enumerate(raw.ch_names)}
    raw.rename_channels(name_map)
    epochs.rename_channels(name_map)

    num_channels = 3
    epochs = epochs.copy().pick_channels(epochs.ch_names[:num_channels])

    return raw, epochs

@pytest.mark.parametrize("feature", features)
def test_complexity_feature_epochs(raw_epochs, feature):
    _, epochs = raw_epochs
    func = fe.FunctionalFeatureRegistry.get(feature)
    result = func(epochs)
    assert hasattr(result, "values")
    assert hasattr(result, "metadata")
    assert isinstance(result.values, np.ndarray)
    # Check shape: (n_epochs, n_channels) or (n_channels,)
    assert result.values.ndim in [1, 2]
    #result.inspect()  # Optional, if you want to check visualization

@pytest.mark.parametrize("feature", features)
def test_complexity_feature_raw(raw_epochs, feature):
    raw, _ = raw_epochs
    func = fe.FunctionalFeatureRegistry.get(feature)
    result = func(raw)
    assert hasattr(result, "values")
    assert hasattr(result, "metadata")
    assert isinstance(result.values, np.ndarray)
    # Should be 1D (channels,) or (channels, 1) after squeeze
    assert result.values.ndim in [1, 2]
    #result.inspect()  # Optional

if __name__ == "__main__":
    import sys
    import pytest

    # List registered features for verification
    for i in features:
        print(i)

    sys.exit(pytest.main([__file__]))
