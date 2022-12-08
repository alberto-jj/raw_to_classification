import os # For path manipulation
import shutil # File manipulation
from mne_bids import print_dir_tree # To show the input/output directories structures inside this example
from sovabids.rules import apply_rules # Apply rules for conversion
from sovabids.convert import convert_them # Do the conversion
from sovabids.datasets import lemon_prepare # Download the dataset
import mne

# Erase PREP files...

exemplar_file = "Y:/datasets/HenryRailo/data/01_eyesOpen.set"
raw=mne.io.read_raw_eeglab(exemplar_file) #inspect line noise freq and channel names

source_path = "Y:/datasets/HenryRailo/data"
bids_path= "Y:/datasets/HenryRailo/bids"
rules_path = "Y:/datasets/HenryRailo/rules.yml"
mapping_path = "Y:/datasets/HenryRailo/mappings.yml"

apply_rules(source_path,bids_path,rules_path,mapping_path)
convert_them(mapping_path)
print_dir_tree(bids_path)