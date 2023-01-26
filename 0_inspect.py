import mne

from eeg_raw_to_classification.utils import load_yaml
datasets = load_yaml('datasets.yml')
DATASET = datasets['FINLAND']

exemplar_file = DATASET['example_file']
raw=mne.io.read_raw_eeglab(exemplar_file) #inspect line noise freq and channel names
