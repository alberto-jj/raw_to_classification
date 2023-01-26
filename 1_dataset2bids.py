from mne_bids import print_dir_tree # To show the input/output directories structures inside this example
from sovabids.rules import apply_rules # Apply rules for conversion
from sovabids.convert import convert_them # Do the conversion

from eeg_raw_to_classification.utils import load_yaml
datasets = load_yaml('datasets.yml')

for dslabel,DATASET in datasets.items():

    source_path = DATASET['sovabids']['paths']['source_path']
    bids_path = DATASET['sovabids']['paths']['bids_path']
    rules = DATASET['sovabids']['rules']
    mapping_path = DATASET['sovabids']['paths']['mapping_path']

    apply_rules(source_path,bids_path,rules,mapping_path)
    convert_them(mapping_path)
    print_dir_tree(bids_path)