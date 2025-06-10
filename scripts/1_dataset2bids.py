import argparse
from mne_bids import print_dir_tree # To show the input/output directories structures inside this example
from sovabids.rules import apply_rules # Apply rules for conversion
from sovabids.convert import convert_them # Do the conversion
import importlib

from eeg_raw_to_classification.utils import load_yaml,get_path

def main(pipeline_file):
    cfg = load_yaml(pipeline_file)
    MOUNT = cfg.get('mount', None)
    datasets = load_yaml(get_path(cfg['datasets_file'],MOUNT))
    this_dataset2bids = cfg.get('1_dataset2bids', {})

    if 'redefine_bidsify' in this_dataset2bids:
        print('Redifining bidsify function from another module')
        import_dict = this_dataset2bids['redefine_bidsify']
        module_name = import_dict['from_this']
        func_name = import_dict['import_that']
        module = importlib.import_module(module_name)
        bidsify = getattr(module, func_name)
        print(f"Using {module_name}.{func_name} for bidsify function")
    else:
        def bidsify(source_path, bids_path, DATASET_CFG, cfg):
            rules = DATASET_CFG['bidsify']['rules'] # this is suppose to be the dictionary of rules
            mappings = apply_rules(source_path, bids_path, rules)
            convert_them(mappings)
            print_dir_tree(bids_path)

    for dslabel, DATASET in datasets.items():
        if DATASET.get('skip', False):
            continue

        bidsify_cfg = DATASET.get('bidsify', None)

        if bidsify_cfg :
            source_path = DATASET['bidsify']['paths']['source_path']
            source_path = get_path(source_path, MOUNT)
            bids_path = DATASET['bidsify']['paths']['bids_path']
            bids_path = get_path(bids_path, MOUNT)

            bidsify(source_path, bids_path, DATASET, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the dataset to BIDS conversion pipeline.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline YAML file')
    args = parser.parse_args()
    main(args.pipeline_file)