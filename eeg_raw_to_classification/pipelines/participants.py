import argparse
import pandas as pd
from eeg_raw_to_classification.utils import load_yaml,get_path
import shutil
import importlib

def default_clean_participants(participants_file, DATASET):
    """
    Default function to clean participants data.
    This function can be overridden by a custom function defined in the pipeline configuration.
    """
    # Add any other default cleaning steps here
    # DO SOMETHING BASED ON DATASET...
    df = pd.read_csv(participants_file, sep='\t', encoding='utf-8')
    return df.copy()

def another_default_clean_participants(participants_file, DATASET):
    """ Another default function to clean participants data.
    This is just an example of how you can define multiple default cleaning functions.
    """


    reader = eval(participants_cfg['reader']['function'])
    reader_args = participants_cfg['reader']['args']
    df = reader(participants_file, **reader_args)

    participants_cfg = DATASET.get('participants', {})
    if 'columns' not in participants_cfg:
        df = df.rename(columns=participants_cfg['columns'])
        columns = list(participants_cfg['columns'].values())
        df = df[columns]

    if 'columns_mapping' in participants_cfg:
        for key, val in participants_cfg['columns_mapping'].items():
            if isinstance(val, dict):
                foo = lambda x: val[x]
            else:
                foo = eval(val)
            df[key] = df[key].apply(foo)
    return df.copy()


def pipeline_participants(pipeline_file):
    cfg = load_yaml(pipeline_file)
    MOUNT = cfg.get('mount', None)
    datasets = load_yaml(get_path(cfg['datasets_file'],MOUNT))
    pipeline_cfg = cfg.get('2_participants', {})

    for dslabel, DATASET in datasets.items():

        if DATASET.get('skip', False):
            continue
        participants_cfg = DATASET.get('participants', {})
        participants_file = participants_cfg['participants_file']
        participants_file = get_path(participants_file, MOUNT)
        outfile = participants_cfg['cleaned_participants']
        outfile = get_path(outfile, MOUNT)

        # create copy if participants_file is the same as outfile
        if participants_file == outfile:
            participants_file2 = participants_file + '.copy'
            shutil.copy(outfile, participants_file2)



        if 'redefine_clean_participants' in pipeline_cfg:
            print('Redifining clean_participants function from another module')
            import_dict = pipeline_cfg['redefine_clean_participants']
            module_name = import_dict['from_this']
            func_name = import_dict['import_that']
            module = importlib.import_module(module_name)
            clean_participants = getattr(module, func_name)
            print(f"Using {module_name}.{func_name} for clean_participants function")
        else:
            clean_participants = default_clean_participants

        df = clean_participants(participants_file, DATASET)
        df.to_csv(outfile, index=False)
        print('Processed dataset:', dslabel, 'saved to:', outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pipeline file.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline YAML file')
    args = parser.parse_args()
    pipeline_participants(args.pipeline_file)