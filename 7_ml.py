#use automl environment
import pandas as pd
import pickle
#import seaborn as sns
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from supervised.automl import AutoML
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json,save_figs_in_html
import glob
from autogluon.tabular import TabularDataset, TabularPredictor
import traceback

PIPELINE = load_yaml(f'pipeline.yml')
datasets = load_yaml(PIPELINE['datasets_file'])

PROJECT = PIPELINE['project']

OUTPUT_DIR = PIPELINE['ml']['path'].replace('%PROJECT%,PROJECT')
os.makedirs(OUTPUT_DIR,exist_ok=True)

fold_path = os.path.join(PIPELINE['scalingAndFolding']['path'])
fold_pattern = os.path.join(fold_path,'**','folds-*.pkl')
foldinstances = glob.glob(fold_pattern,recursive=True)
#foldinstances = [x for x in foldinstances if 'folding' in x]
scalingAndFolding_path = PIPELINE['scalingAndFolding']['path']
ml_path=PIPELINE['ml']['path']
for _foldpath in foldinstances:

    foldcomb = os.path.basename(_foldpath).split('-')[1].split('.')[0]
    #foldsavepath = os.path.join(OUTPUT_DIR,foldcomb) # this is already in _foldpath
    foldsavepath=os.path.dirname(_foldpath.replace(scalingAndFolding_path,ml_path))
    os.makedirs(foldsavepath,exist_ok=True)

    for mlmodel,mlparams in PIPELINE['ml']['models'].items():
        savepath = os.path.join(foldsavepath,mlmodel)
        os.makedirs(savepath,exist_ok=True)

        try:
            _foldinstance = pickle.load(open(_foldpath,'rb'))

            # Make superdataset
            dfX_train = []
            dfY_train = []
            dfX_test = []
            dfY_test = []
            df_train = []
            df_test = []
            foldnums = list(_foldinstance.keys())
            for foldnum,data in _foldinstance.items():
                dfX_train.append(data['X_train'])
                dfY_train.append(data['Y_train'])
                dfX_test.append(data['X_test'])
                dfY_test.append(data['Y_test'])
                df_train.append(data['df_train'])
                df_test.append(data['df_test'])
            dfX_train = pd.concat(dfX_train,ignore_index=True)
            dfX_train['phase']='train'
            dfY_train = pd.concat(dfY_train,ignore_index=True)
            dfY_train['phase']='train'
            dfX_test = pd.concat(dfX_test,ignore_index=True)
            dfX_test['phase']='test'
            dfY_test = pd.concat(dfY_test,ignore_index=True)
            dfY_test['phase']='test'
            df_train = pd.concat(df_train,ignore_index=True)
            df_train['phase']='train'
            df_test = pd.concat(df_test,ignore_index=True)
            df_test['phase']='test'

            drops=['phase']
            dfX = pd.concat([dfX_train,dfX_test],ignore_index=True).drop(drops,axis=1)
            dfY= pd.concat([dfY_train,dfY_test],ignore_index=True).drop(drops,axis=1)
            df = pd.concat([df_train,df_test],ignore_index=True)

            assert len(dfX)==len(dfY)==len(df)


            # TODO: Add feature engineering/selection step here from previous insights (e.g. featurewiz)
            # e.g. configure feature selection from the pipeline.yml

            # Make indexes for folds
            fold_tuples = []
            for foldnum in foldnums:
                # train,test
                fold_tuples.append((df[(df['foldIter']==foldnum) & (df['phase']=='train')].index,df[(df['foldIter']==foldnum) & (df['phase']=='test')].index))
                if 'folding' in foldcomb:
                    assert foldnum not in df[(df['foldIter']==foldnum) & (df['phase']=='train')]['foldSet'].unique()

            mlmethod = mlparams['method']
            if mlmethod == 'AutoMLjar':
                # https://supervised.mljar.com/api/
                AutoML_ = AutoML(**mlparams['init'] ,results_path=savepath)
                AutoML_.fit(dfX,dfY,cv=fold_tuples)

            if mlmethod == 'AutoGluon':
                #https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html

                leads_train = []
                leads_test = []
                for foldnum,data in _foldinstance.items():
                    this_path = os.path.join(savepath,f'fold_{foldnum}')
                    this_path_log = os.path.join(savepath,f'fold_{foldnum}_logs')
                    # assume 1 single target
                    label=dfY_test.columns[0]
                    predictor = TabularPredictor(label=label,**mlparams['init'],path=this_path,log_file_path=this_path_log)

                    # We should fix the fact that folds would be reduced by cutting out tuning_data
                    # See: https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html#tabularpredictor-fit
                    # If tuning_data = None, fit() will automatically hold out some random validation examples from train_data.
                    dfX_train = data['X_train']
                    dfY_train = data['Y_train']
                    dfX_train[label]=dfY_train[label] # Add target as it expects full dataframe

                    dfX_test = data['X_test']
                    dfY_test = data['Y_test']
                    dfX_test[label]=dfY_test[label]

                    predictor.fit(dfX_train, **mlparams['fit'])

                    leadf_train=predictor.leaderboard(dfX_train)
                    leadf_train['fold']=foldnum

                    predictor.evaluate(dfX_test, silent=True)
                    leadf_test = predictor.leaderboard(dfX_test)
                    leadf_test['fold']=foldnum

                    leads_train.append(leadf_train)
                    leads_test.append(leadf_test)
                
                leads_train = pd.concat(leads_train,ignore_index=True)
                leads_test = pd.concat(leads_test,ignore_index=True)
                leads_train.to_csv(os.path.join(savepath,'leaderboard_train.csv'))
                leads_test.to_csv(os.path.join(savepath,'leaderboard_test.csv'))

                def agg_func(x):
                    if x.dtype == object:
                        return x.mode()[0]  # Return the most frequent value for strings
                    elif x.dtype == bool:
                        return x.astype(int).mean()  # Convert bools to 0/1 and take mean
                    else:
                        return x.mean()  # Return the mean for numerics

                leadf_train = leads_train.drop(['fold'],axis=1).groupby('model').agg(agg_func).reset_index()
                leadf_test = leads_test.drop(['fold'],axis=1).groupby('model').agg(agg_func).reset_index()
                leadf_train.to_csv(os.path.join(savepath,'leaderboard_train_aggfolds.csv'))
                leadf_test.to_csv(os.path.join(savepath,'leaderboard_test_aggfolds.csv'))
        except Exception as e:
            print(f'Error in {mlmodel}-{foldcomb} - {e}')
            msg=traceback.format_exc()
            save_dict_to_json(os.path.join(savepath,f'error_ml-{mlmodel}_data-{foldcomb}.json'),{'error':msg})
