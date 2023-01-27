import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict
import itertools
import pandas as pd

pipeline_name = 'aggregate'

datasets = load_yaml('datasets.yml')
cfg = load_yaml('pipeline.yml')
OUTPUT = cfg['aggregate']['path']
id_splitter = cfg['aggregate']['id_splitter']
os.makedirs(OUTPUT,exist_ok=True)
ALL = []
 
for dslabel, DATASET in datasets.items():
    perfeature =[]
    participants_file = DATASET['cleaned_participants']
    participants = pd.read_csv(participants_file)
    def parfun(query,field):
        try:
            sub = int(query)
        except:
            sub = query
        idx = participants[participants['subject']==sub]
        assert idx.shape[0]==1 #unique
        return idx[field].item()
    for feature in cfg['aggregate']['features']:
        pattern = os.path.join(DATASET.get('bids_root', None),'derivatives','features',f'**/*_{feature}.npy').replace('\\','/')
        eegs = glob.glob(pattern,recursive=True)
        #output =os.path.join(DATASET.get('bids_root', None),'derivatives',pipeline_name,f'{feature}.csv')
        #os.makedirs(os.path.dirname(output),exist_ok=True)
        dict_list=[]

        for eeg_file in eegs:
            suffix = os.path.basename(eeg_file).split('_')[-1].split('.')[0] +'.'
            dict_list += get_output_dict(eeg_file,'WIDE',DATASET['dataset_label'],suffix)

        df = pd.DataFrame(dict_list)
        df.insert(loc=0,column='id',value=df['dataset']+id_splitter+df['subject']+id_splitter+df['task'])
        for field in ['group','age','sex']:
            auxdf = df['subject'].apply(lambda x: parfun(x,field))
            df.insert(loc=1,column=field,value=auxdf)
        perfeature.append(df)
    ALL+=perfeature

df = ALL[0]
for a in ALL[1:]:
    df = pd.merge(df, a, on="id",validate='1:1',suffixes=(None,'_y'))

del df['task_y']
del df['subject_y']
del df['dataset_y']
df.to_csv(os.path.join(OUTPUT,'multidataset.csv'),index=False)

print('ok')