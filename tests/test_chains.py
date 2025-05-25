import eeg_raw_to_classification.feature_extraction as fe
from eeg_raw_to_classification.feature_extraction import run_feature
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



# BasicPrep



# SpectrumMultitaper
# SpectrumMultitaperEpochsAverage
# AbsBandPower
# AbsBandPowerEpochsAverage
# FooofFromEpochsAverage
# BandPowerRatiosEpochsAverage
if __name__ == "__main__":
    import sys
    import pytest


    for i in fe.ChainFeatureRegistry.list():
        print(i)


    # Run only this file
    sys.exit(pytest.main([__file__]))
