import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from eeg_raw_to_classification.aggregates import *


pipeline_name = 'aggregate'

datasets = load_yaml('datasets.yml')
cfg = load_yaml('pipeline.yml')
OUTPUT = cfg['aggregate']['path']
filename = cfg['aggregate']['filename']
id_splitter = cfg['aggregate']['id_splitter']
os.makedirs(OUTPUT,exist_ok=True)
ALL = []

# Find Common Montage
MONTAGES = []
for dslabel, DATASET in datasets.items():
    MONTAGES.append(DATASET['ch_names'])

common = set(MONTAGES[0])
for montage in MONTAGES[1:]:
    common = common.intersection(set(montage))
save_dict_to_json(os.path.join(OUTPUT,'common_montage.txt'),{'common_montage':list(common)})
for dslabel, DATASET in datasets.items():
    perfeature =[]
    participants_file = DATASET['cleaned_participants']
    participants = pd.read_csv(participants_file)
    def parfun(query,field):
        try:
            sub = int(query) # im not conviced this is a good idea
        except:
            sub = query
        idx = participants[participants['subject']==sub]
        assert idx.shape[0]==1 #unique
        return idx[field].item()
    for feature in cfg['aggregate']['features_list']:
        pattern = os.path.join(DATASET.get('bids_root', None),'derivatives','features',f'**/*_{feature}.npy').replace('\\','/')
        eegs = glob.glob(pattern,recursive=True)
        #output =os.path.join(DATASET.get('bids_root', None),'derivatives',pipeline_name,f'{feature}.csv')
        #os.makedirs(os.path.dirname(output),exist_ok=True)
        dict_list=[]
        foodict = cfg['aggregate']['feature_return'][feature]
        foo = eval(foodict['return_function'].replace('eval%',''))

        for eeg_file in eegs:
            suffix = os.path.basename(eeg_file).split('_')[-1].split('.')[0] +'.'
            dict_list += get_output_dict(eeg_file,'WIDE',DATASET['dataset_label'],suffix,agg_fun=None)


        df = pd.DataFrame(dict_list)

        # Get metadata of that feature using last eeg_file
        metafeature = np.load(eeg_file,allow_pickle=True).item()['metadata']
        idx_space = metafeature['order'].index('spaces') + 1 # increase 1 because col name includes type at start

        # Remove non commmon features
        derivative_cols = [x.split('.')[idx_space] if '.' in x else 'IGNORE' for x in df.columns ] # All derivatives must have a dot at least, to indicate the type
        common = common.union({'IGNORE'})
        keep = [True if derivative_cols[i] in common else False for i,x in enumerate(derivative_cols)]
        df = df.iloc[:,keep]

        df.insert(loc=0,column='id',value=df['dataset']+id_splitter+df['subject']+id_splitter+df['task'])
        for field in ['group','age','sex']: #TODO: maybe this should be configured from outside 
            auxdf = df['subject'].apply(lambda x: parfun(x,field))
            df.insert(loc=1,column=field,value=auxdf)
        perfeature.append(df)
    df = perfeature[0]
    for a in perfeature[1:]:
        omit_cols = a.drop(['age','dataset','group','sex','subject','task'],axis='columns') #TODO: maybe this should be configured from outside
        df = pd.merge(df, omit_cols, on="id",validate='1:1',suffixes=(None,'_y'))

    ALL.append(df)

# Verify Features
num_features = [x.shape[1] for x in ALL]
assert len(set(num_features)) == 1
cols = [x.columns for x in ALL]
colset = set(cols[0])
for x in cols[1:]:
    colset = colset.intersection(set(x))
assert set(cols[0])==colset

# Force same order of columns
# find first derivative column
new_orders = []
for i in range(len(cols)):
    dots = ['.' in x for x in cols[i]]
    first_der = dots.index(True)
    feats = list(cols[i][first_der:])
    feats.sort()
    feats = list(cols[i][:first_der])+feats
    new_orders.append(feats)
ALL = [x[new_orders[i]] for i,x in enumerate(ALL)]

# Verify order
cols = [x.columns for x in ALL]
ref = cols[0]
equal_cols = []
for x in cols[1:]:
    equal_cols.append(all(ref==x))
assert all(equal_cols)

# Concatenate
df = pd.concat(ALL,axis=0,ignore_index=True)
df.to_csv(os.path.join(OUTPUT,filename),index=False)

print('ok')