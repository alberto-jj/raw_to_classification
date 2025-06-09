import pytest
import mne
import numpy as np

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

def test_binarizer_on_bandpower(raw_sample):
    # Band spectrum (for a nice, nontrivial test case)
    spectrum = pf.functional_spectrum_feature(raw_sample, method="multitaper", mne_kwargs={'adaptive': False})
    bands = {
        'delta': [0.5, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 45]
    }
    bandpower = pf.functional_bandspectrum_feature(spectrum, bands=bands, relative=False)

    # Binarize along spaces (channels)
    binarized = pf.functional_binarize_along_axis_feature(bandpower, axisname='spaces', threshold_fun='median')
    assert hasattr(binarized, "values")
    assert hasattr(binarized, "metadata")
    assert np.issubdtype(binarized.values.dtype, np.integer)
    assert set(np.unique(binarized.values)) <= {0, 1}
    binarized.inspect()  # Optionally visualize

def test_binarizer_on_times(raw_sample):
    spectrum = pf.functional_spectrum_feature(raw_sample, method="multitaper", mne_kwargs={'adaptive': False})
    # Binarize spectrum along the time/frequency axis
    binarized = pf.functional_binarize_along_axis_feature(spectrum, axisname='frequencies', threshold_fun='median')
    assert hasattr(binarized, "values")
    assert hasattr(binarized, "metadata")
    assert np.issubdtype(binarized.values.dtype, np.integer)
    assert set(np.unique(binarized.values)) <= {0, 1}
    binarized.inspect()

def test_elementwise_abs(raw_sample):
    spectrum = pf.functional_spectrum_feature(raw_sample, method="multitaper", mne_kwargs={'adaptive': False})
    # Test elementwise with numpy abs
    abs_result = pf.functional_elementwise_feature(spectrum, fun=np.abs)
    assert hasattr(abs_result, "values")
    assert hasattr(abs_result, "metadata")
    assert np.all(abs_result.values >= 0)
    abs_result.inspect()

def test_elementwise_lambda(raw_sample):
    spectrum = pf.functional_spectrum_feature(raw_sample, method="multitaper", mne_kwargs={'adaptive': False})
    # Test elementwise with a lambda that returns 1 for all values > 0.5, else 0
    bin_result = pf.functional_elementwise_feature(spectrum, fun=lambda x: int(x > 0.5), newtype="int")
    assert hasattr(bin_result, "values")
    assert hasattr(bin_result, "metadata")
    assert set(np.unique(bin_result.values)) <= {0, 1}
    bin_result.inspect()

def test_elementwise_eval_lambda(raw_sample):
    spectrum = pf.functional_spectrum_feature(raw_sample, method="multitaper", mne_kwargs={'adaptive': False})
    # Test elementwise with a string lambda using eval%
    bin_result = pf.functional_elementwise_feature(spectrum, fun="eval%lambda x: int(x > 0.5)", newtype="int")
    assert hasattr(bin_result, "values")
    assert hasattr(bin_result, "metadata")
    assert set(np.unique(bin_result.values)) <= {0, 1}
    bin_result.inspect()




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
