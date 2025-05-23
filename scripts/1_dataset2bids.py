import argparse
from mne_bids import print_dir_tree # To show the input/output directories structures inside this example
from sovabids.rules import apply_rules # Apply rules for conversion
from sovabids.convert import convert_them # Do the conversion

from eeg_raw_to_classification.utils import load_yaml,get_path

def main(pipeline_file):
    cfg = load_yaml(pipeline_file)
    MOUNT = cfg.get('mount', None)
    datasets = load_yaml(get_path(cfg['datasets_file'],MOUNT))

    for dslabel, DATASET in datasets.items():
        if DATASET.get('skip', False):
            continue

        if 'sovabids' in DATASET:
            source_path = DATASET['sovabids']['paths']['source_path']
            source_path = get_path(source_path, MOUNT)
            bids_path = DATASET['sovabids']['paths']['bids_path']
            bids_path = get_path(bids_path, MOUNT)
            rules = DATASET['sovabids']['rules'] # this is suppose to be the dictionary of rules

            mappings = apply_rules(source_path, bids_path, rules)
            convert_them(mappings)
            print_dir_tree(bids_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the dataset to BIDS conversion pipeline.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline YAML file')
    args = parser.parse_args()
    main(args.pipeline_file)