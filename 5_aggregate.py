import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




cfg = load_yaml(f'pipeline.yml')
datasets = load_yaml(cfg['datasets_file'])
PROJECT = cfg['project']
OUTPUTBASE = cfg['aggregate']['path'].replace('%PROJECT%',PROJECT)
csvfilename = cfg['aggregate']['filename']
id_splitter = cfg['aggregate']['id_splitter']
os.makedirs(OUTPUTBASE,exist_ok=True)


for agg_cfg_label in cfg['aggregate']['feature_aggregate_list']:
    agg_cfg = cfg['aggregate']['aggregate_cfgs'][agg_cfg_label]
    OUTPUT = os.path.join(OUTPUTBASE,agg_cfg_label)
    os.makedirs(OUTPUT,exist_ok=True)
    ALL_INHOMOGENEUS = []
    COMMON_FEATURES = []


    for dslabel, DATASET in datasets.items():
        if DATASET.get('skip',False):
            continue

        perfeature =[]
        participants_file = DATASET['cleaned_participants']
        participants = pd.read_csv(participants_file)
        def parfun(query,field):
            try:
                sub = int(query) # im not conviced this is a good idea
            except:
                sub = query
            try:
                idx = participants[participants['subject']==sub]
            except:
                idx = participants[participants['number']==sub] # this is very hardcoded, this should be a configuration
            assert idx.shape[0]==1 #unique
            return idx[field].item()
        for feature in agg_cfg['feature_list']:
            print(f'Processing {feature} in {dslabel}')
            featfolder=agg_cfg['feature_folder']
            foodict = cfg['aggregate']['feature_return'][feature]
            filesuffix = foodict['file_suffix']
            pattern = os.path.join(DATASET.get('bids_root', None),'derivatives',featfolder,f'**/*_{filesuffix}.npy').replace('\\','/')
            eegs = glob.glob(pattern,recursive=True)
            #output =os.path.join(DATASET.get('bids_root', None),'derivatives',pipeline_name,f'{feature}.csv')
            #os.makedirs(os.path.dirname(output),exist_ok=True)
            dict_list=[]
            foodict = cfg['aggregate']['feature_return'][feature]
            foo = eval(foodict['return_function'].replace('eval%',''))

            #dict_list_long = []
            for eeg_file in eegs:
                suffix = os.path.basename(eeg_file).split('_')[-1].split('.')[0] +'.'
                desired_label = feature + '.' # dot is important for combination format
                dict_list += get_output_dict(eeg_file,'WIDE',DATASET['dataset_label'],desired_label,agg_fun=foo,keyvalformat=True)
                #dict_list_long += get_output_dict(eeg_file,'LONG',DATASET['dataset_label'],desired_label[:-1],agg_fun=foo,keyvalformat=True) # no dot
                #TODO per aggregate save feature ontology --> i guess this mean saving descriptions of the features??? dont remember what i meant...

            if len(dict_list)==0:
                print(f'No files found for {feature} in {dslabel}')
                continue
            df = pd.DataFrame(dict_list)
            #df_long=pd.DataFrame(dict_list_long)
            # Get metadata of that feature using last eeg_file
            metafeature = np.load(eeg_file,allow_pickle=True).item()['metadata']
            idx_space = metafeature['order'].index('spaces') + 1 # increase 1 because col name includes feature type at start


            df.insert(loc=0,column='id',value=df['dataset']+id_splitter+df['subject']+id_splitter+df['task'])
            for field in ['group','age','sex']: #TODO: maybe this should be configured from outside 
                auxdf = df['subject'].apply(lambda x: parfun(x,field))
                df.insert(loc=1,column=field,value=auxdf)
            # why do we need to drop these cols
            # omit_cols=['age','dataset','group','sex','subject','task']
            # df=df.drop(omit_cols,axis='columns')

            # Identify derivative(feature) columns
            derivative_cols = [x if '.' in x else 'IGNORE' for x in df.columns ] # All derivatives must have a dot at least, to indicate the type

            perfeature.append(df)

        if len(perfeature)==0:
            print(f'No features found for {dslabel}')
            continue
        df = perfeature[0]
        for a in perfeature[1:]:
            df = pd.merge(df, a, on="id", validate='1:1', suffixes=(None, '_y'))
            # verify overlapping columns have the same info
            overlapping_cols = [col for col in df.columns if col.endswith('_y')]
            for col in overlapping_cols:
                col_name = col[:-2]
                if df[col_name].equals(df[col]):
                    df=df.drop(col, axis=1, inplace=False)

        ALL_INHOMOGENEUS.append(df)

    # Verify Features
    column_sets = [x.columns for x in ALL_INHOMOGENEUS]
    common=set(column_sets[0])
    for feature_set in column_sets[1:]:
        common = common.intersection(set(feature_set))
    
    ALL=[]
    for df,this_set in zip(ALL_INHOMOGENEUS,column_sets):
        ALL.append(df[list(common)])


    num_features = [x.shape[1] for x in ALL]
    assert len(set(num_features)) == 1
    cols = [x.columns for x in ALL]
    colset = set(cols[0])
    for x in cols[1:]:
        colset = colset.intersection(set(x))
    assert set(cols[0])==colset

    # Force same order of columns
    # find first derivative column
    feature_cols = [x for x in common if '.' in x]
    non_feature_cols = [x for x in common if not '.' in x]
    feature_cols.sort()
    non_feature_cols.sort()
    new_order=non_feature_cols+feature_cols
    ALL = [x[new_order] for i,x in enumerate(ALL)]

    # Verify order
    cols = [x.columns for x in ALL]
    ref = cols[0]
    equal_cols = []
    for x in cols[1:]:
        equal_cols.append(all(ref==x))
    assert all(equal_cols)

    save_dict_to_json(os.path.join(OUTPUT,'common_cols.txt'),{'common_cols':new_order})

    # Concatenate
    df = pd.concat(ALL,axis=0,ignore_index=True)
    # drop rows with nans
    df.to_csv(os.path.join(OUTPUT,csvfilename+'@raw.csv'),index=False)

    print('ok')