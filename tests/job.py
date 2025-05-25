import mne

print('MNE version:', mne.__version__)

from eeg_raw_to_classification import features as feat
from eeg_raw_to_classification.utils import load_yaml, save_dict_to_json

print(dir(feat))