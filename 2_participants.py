import pandas as pd
from eeg_raw_to_classification.utils import load_yaml
import pandas as pd

datasets = load_yaml('datasets.yml')

for dslabel,DATASET in datasets.items():

    datafile = DATASET['participants_file']
    outfile = DATASET['cleaned_participants']
    if '.csv' in datafile:
        df = pd.read_csv(datafile)
    else:
        df = pd.read_excel(datafile)

    df = df.rename(columns=DATASET['columns'])
    columns = list(DATASET['columns'].values())
    df = df[columns]

    # dataset specific
    if 'df_transform' in DATASET:
        exec(DATASET['df_transform'])

    if 'columns_mapping' in DATASET:

        for key,val in DATASET['columns_mapping'].items():
            if isinstance(val,dict):
                foo = lambda x:val[x]
            else:
                foo = eval(val)
            df[key]=df[key].apply(foo)

    df.to_csv(outfile,index=False)
    print('ok')