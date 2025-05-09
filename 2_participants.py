import argparse
import pandas as pd
from eeg_raw_to_classification.utils import load_yaml
import shutil

def main(pipeline_file):
    cfg = load_yaml(pipeline_file)
    datasets = load_yaml(cfg['datasets_file'])

    for dslabel, DATASET in datasets.items():

        if DATASET.get('skip', False):
            continue

        datafile = DATASET['participants_file']
        outfile = DATASET['cleaned_participants']

        # create copy if datafile is the same as outfile
        if datafile == outfile:
            datafile2 = datafile + '.copy'
            shutil.copy(outfile, datafile2)

        reader = eval(DATASET['reader']['function'])
        reader_args = DATASET['reader']['args']
        df = reader(datafile, **reader_args)
        scope = {
            'df': df,
            'DATASET': DATASET,
            'datafile': datafile,
            'outfile': outfile
        }
        # dataset specific
        if 'df_transform' in DATASET:
            exec(DATASET['df_transform'], None, scope)
            df = scope['df']

        df = df.rename(columns=DATASET['columns'])
        columns = list(DATASET['columns'].values())
        df = df[columns]

        if 'columns_mapping' in DATASET:
            for key, val in DATASET['columns_mapping'].items():
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