import mne

# pip install pympler

#TODO: make tests using epochs and raw files from sample datasets of mne
path = r"Y:\datasets\epilepsy\bids\derivatives\defaultprep\sub-S001\run-1P\sub-S001_run-1P_task-SZ_acq-EZ0_outcome-SF_desc-reject_epo.fif"

epochs = mne.read_epochs(path, verbose=False)
print(epochs)

from eeg_raw_to_classification.features_extraction.spectralFunctional import functional_spectrum_feature

output = functional_spectrum_feature(epochs, method='welch', mne_kwargs={'n_fft': 256, 'n_overlap': 128, 'average': 'mean'}, label='test')
epochs.get_data().shape
output.values.shape
output.metadata.axes
output.metadata.order
sample_data_folder = mne.datasets.sample.data_path(download=True)
sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file,preload=True,verbose=False)



print(raw)

raw3 = mne.io.read_raw_fif(sample_data_raw_file,preload=False,verbose=False)


from pympler import asizeof


from eeg_raw_to_classification.features_extraction import process_feature, FunctionalFeatureRegistry, ChainFeatureRegistry
from eeg_raw_to_classification.features_extraction import SpectrumMultitaperAverage

output = process_feature(epochs, SpectrumMultitaperAverage, path,inspect_only=False, chain_feature_registry=ChainFeatureRegistry, functional_feature_registry=FunctionalFeatureRegistry)
output = process_feature(epochs, SpectrumMultitaperAverage, None,inspect_only=False, chain_feature_registry=ChainFeatureRegistry, functional_feature_registry=FunctionalFeatureRegistry)
output = process_feature(epochs, SpectrumMultitaperAverage, path,inspect_only=True,  chain_feature_registry=ChainFeatureRegistry, functional_feature_registry=FunctionalFeatureRegistry)
output = process_feature(epochs, SpectrumMultitaperAverage, None,inspect_only=True,  chain_feature_registry=ChainFeatureRegistry, functional_feature_registry=FunctionalFeatureRegistry)

def examine_object(obj):
    for attr_name in dir(obj):
        if attr_name.startswith('_'):  # skip private attrs if desired
            continue
        try:
            attr = getattr(raw, attr_name)
            size = asizeof.asizeof(attr)
            print(f"{attr_name:<25} {size/1024:.2f} KB")
        except Exception as e:
            print(f"{attr_name:<25} [ERROR] {e}")

examine_object(raw3)
examine_object(raw3.info)


for k in raw3.info:
    try:
        size = asizeof.asizeof(raw3.info[k])
        print(f"{k:<20} {size / 1024:.2f} KB")
    except Exception as e:
        print(f"{k:<20} [ERROR] {e}")

raw3.info['hpi_meas']


import pickle

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(f"Object saved to {name}")

def load_obj(name):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {name}")
    return obj

save_obj(raw3, 'raw3.pkl')