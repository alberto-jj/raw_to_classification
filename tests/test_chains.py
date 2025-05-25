import pytest
import mne

import eeg_raw_to_classification.feature_extraction as fe


# ------------ Fixtures ------------

def raw_epochs2():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False).crop(tmax=5)
    raw = raw.pick_types(eeg=True, exclude="bads")
    num_channels = 3
    raw = raw.copy().pick_channels(raw.ch_names[:num_channels])
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=True, verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    name_map = {ch: montage.ch_names[i] for i, ch in enumerate(raw.ch_names)}
    raw.rename_channels(name_map)
    epochs.rename_channels(name_map)
    return raw, epochs

@pytest.fixture(scope="module")
def raw_epochs():
    return raw_epochs2()



# ------------ Parameter Space ------------

chain_names = [
    "BasicPrep",  # raw only
    "SpectrumMultitaper",
    "SpectrumMultitaperEpochsAverage",
    "AbsBandPower",
    "AbsBandPowerEpochsAverage",
    "FooofFromEpochsAverage",
    "BandPowerRatiosEpochsAverage",
]

inspect_modes = [False, True]
bidspaths = [None, "data/test/sub-test_mne.fif"]


# ------------ Tests ------------

@pytest.mark.parametrize("inspect_only", inspect_modes)
@pytest.mark.parametrize("file_bidspath", bidspaths)
@pytest.mark.parametrize("name", chain_names)
def test_chain_features(raw_epochs, inspect_only, file_bidspath, name):
    raw, epochs = raw_epochs

    print(f"\nRunning chain: {name} | inspect_only={inspect_only} | bidspath={file_bidspath}")

    feature_config = getattr(fe, name)

    if name == "BasicPrep":
        feature_config.chain[0]["args"]['mne_kwargs']['duration'] = 1
        feature_config.chain[0]["args"]['mne_kwargs']['overlap'] = 0.5
        feature_config.chain[0]["args"]['label'] = "Epochs1sOverlap05s"
        result = fe.run_feature(raw, feature_config, inspect_only=inspect_only, file_bidspath=file_bidspath)
    else:
        result = fe.run_feature(epochs, feature_config, inspect_only=inspect_only, file_bidspath=file_bidspath)

    if not inspect_only:
        assert result is not None, f"{name} returned None unexpectedly"
        result.inspect()

if __name__ == "__main__":
    import sys
    import pytest
    
    # if one failes you can run it alone like this
    # name="BandPowerRatiosEpochsAverage"
    # inspect_only=False
    # file_bidspath='sub-test_mne.fif'
    # feature_config = getattr(fe, name)
    # raw_epochs = raw_epochs2()
    # raw, epochs = raw_epochs
    # fe.run_feature(epochs, feature_config, inspect_only=inspect_only, file_bidspath=file_bidspath)
    for i in fe.FunctionalFeatureRegistry.list():
        print(i)

    for i in fe.ChainFeatureRegistry.list():
        print(i)


    # Run only this file
    sys.exit(pytest.main([__file__]))
