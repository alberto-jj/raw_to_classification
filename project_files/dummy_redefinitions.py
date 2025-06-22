import os
import scipy.io as sio
import mne
import numpy as np
import glob
import scipy
from mne.io import read_raw
from mne import read_epochs
import pandas as pd
from sovabids.parsers import parse_from_placeholder
from mne_bids import BIDSPath, write_raw_bids
import traceback
import pdb
from pprint import pprint
from eeg_raw_to_classification.pipelines.dataset2bids import sova_bidsify as bidsify
from eeg_raw_to_classification.pipelines.participants import default_clean_participants
from eeg_raw_to_classification.pipelines.preprocessing import prepare