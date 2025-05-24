import mne
from sys import getsizeof
# or use pympler.asizeof.asizeof(attr) for a more accurate size
# pip install pympler
# from pympler import asizeof.asizeof as getsizeof

def examine_object(obj):
    for attr_name in dir(obj):
        # if attr_name.startswith('_'):  # skip private attrs if desired
        #     continue
        try:
            attr = getattr(raw, attr_name)
            size = getsizeof(attr)
            print(f"{attr_name:<25} {size/1024:.2f} KB")
        except Exception as e:
            print(f"{attr_name:<25} [ERROR] {e}")

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



sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False, preload=True).crop(tmax=60)
events = mne.find_events(raw, stim_channel="STI 014")
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7)

raw_no_data = mne.io.read_raw_fif(sample_data_raw_file,preload=False,verbose=False)
examine_object(raw_no_data)
examine_object(raw)