import argparse
import pandas as pd
from eeg_raw_to_classification.utils import load_yaml,get_path
import shutil

def main(pipeline_file):
    cfg = load_yaml(pipeline_file)
    MOUNT = cfg.get('mount', None)
    datasets = load_yaml(get_path(cfg['datasets_file'],MOUNT))

    for dslabel, DATASET in datasets.items():

        if DATASET.get('skip', False):
            continue
        participants_cfg = DATASET.get('participants', {})
        datafile = participants_cfg['participants_file']
        datafile = get_path(datafile, MOUNT)
        outfile = participants_cfg['cleaned_participants']
        outfile = get_path(outfile, MOUNT)

        # create copy if datafile is the same as outfile
        if datafile == outfile:
            datafile2 = datafile + '.copy'
            shutil.copy(outfile, datafile2)

        reader = eval(participants_cfg['reader']['function'])
        reader_args = participants_cfg['reader']['args']
        df = reader(datafile, **reader_args)
        scope = {
            'df': df,
            'DATASET': DATASET,
            'datafile': datafile,
            'outfile': outfile
        }
        # dataset specific
        if 'df_transform' in  participants_cfg:
            exec(participants_cfg['df_transform'], None, scope)
            df = scope['df']

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

        df.to_csv(outfile, index=False)
        print('Processed dataset:', dslabel, 'saved to:', outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pipeline file.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline YAML file')
    args = parser.parse_args()
    main(args.pipeline_file)