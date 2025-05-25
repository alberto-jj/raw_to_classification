import pytest
import mne

import eeg_raw_to_classification.feature_extraction as pf
import eeg_raw_to_classification.feature_extraction as fe


@pytest.fixture(scope="module")
def raw_sample():
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False).crop(tmax=60)

    raw = raw.pick_types(eeg=True, exclude="bads")

    # Assign standard names
    montage = mne.channels.make_standard_montage('standard_1020')
    name_map = {ch: montage.ch_names[i] for i, ch in enumerate(raw.ch_names)}
    raw.rename_channels(name_map)

    return raw


def test_band_ratios_on_raw(raw_sample):
    # Step 1: Spectrum
    spectrum = pf.functional_spectrum_feature(raw_sample, method="multitaper", mne_kwargs={'adaptive': False})
    spectrum.inspect()

    # Step 2: Band spectrum
    bands = {
        'delta': [0.5, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 45]
    }
    bandpower = pf.functional_bandspectrum_feature(spectrum, bands=bands, relative=False)
    bandpower.inspect()

    # Step 3: Ratio across bands
    ratios = pf.functional_ratio_feature(bandpower, ratio_axis='bands')
    ratios.inspect()

def test_spectrum_space_aggregation(raw_sample):
    # Step 1: Compute spectrum
    spectrum = pf.functional_spectrum_feature(raw_sample, method="multitaper", mne_kwargs={'adaptive': False})
    spectrum.inspect()

    # Step 2: Aggregate across 'spaces' (channels)
    aggregated = pf.functional_aggregate_feature(spectrum, axisname="spaces", fun="mean")
    aggregated.inspect()
    
if __name__ == "__main__":
    import sys
    import pytest


    for i in fe.FunctionalFeatureRegistry.list():
        print(i)

    for i in fe.ChainFeatureRegistry.list():
        print(i)


    # Run only this file
    sys.exit(pytest.main([__file__]))
