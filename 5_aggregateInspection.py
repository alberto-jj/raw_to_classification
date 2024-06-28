import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import traceback
import pickle

datasets = load_yaml('datasets.yml')
cfg = load_yaml('pipeline.yml')
OUTPUTROOT = 'data/viz'#cfg['aggregate']['path']
# csvfilename = cfg['aggregate']['filename']
# id_splitter = cfg['aggregate']['id_splitter']

#aggregate_file = r'data\aggregate\FeaturesChannels@30@prep-defaultprep\aggregate.csv'

#load pkl
pklpattern ='data\scalingAndFolding\**\scaling_effect.pkl'
pklfiles = glob.glob(pklpattern,recursive=True)

for pklfile in pklfiles:
    foldertree=os.path.dirname(pklfile)
    OUTPUTBASE= os.path.join(OUTPUTROOT,foldertree)


    os.makedirs(OUTPUTBASE,exist_ok=True)


    _foldinstance = pickle.load(open(pklfile,'rb'))
    _foldinstance.keys()
    _foldinstance['after(unseen-in-all-folds)'].keys()
    df = _foldinstance['after(unseen-in-all-folds)']['df']

    # wide to long

    id_vars=[x for x in df.columns if not x.startswith('feature')]
    df_long = pd.melt(df,id_vars=id_vars,var_name='feature',value_name='value')
    # find all extra cols
    featsubcols= df_long['feature'].apply(lambda x: '.'.join([y.split('-')[0] for y in x.split('.')]) if x.startswith('feature') else x)

    extra_cols = featsubcols.unique().tolist()
    extra_cols= [x for x in extra_cols if 'feature' in x]
    extra_cols= [x.split('.') for x in extra_cols]
    #flatten
    extra_cols = list(set(list(itertools.chain(*extra_cols)))-set(['feature']))

    def get_val_bidslike(x,key,returnX=False,delim='.',keyvaldelim='-',valstr=''):
        if '.' in x:
            keyvals = x.split(delim)
            for keyval in keyvals:
                if key in keyval:
                    return valstr+keyval.split(keyvaldelim)[1]
        else:
            if returnX:
                return x
            else:
                return None
    for col in extra_cols:
        df_long['axis-'+col] = df_long['feature'].apply(lambda x: get_val_bidslike(x,col))

    df_long['feature'] = df_long['feature'].apply(lambda x: get_val_bidslike(x,'feature',returnX=True,valstr='feature-'))


    # var_ vs Feature (flat design, no aggregation)

    feature_cols= [col for col in df.columns if col.startswith('feature')] + ['age','sex','group']

    for var_ in ['dataset','age','sex','group']:
        vizout= os.path.join(OUTPUTBASE,f'{var_}_feature_flat')
        os.makedirs(vizout,exist_ok=True)
        for feature in feature_cols:
            try:
                fig,axes= plt.subplots(1,1,figsize=(15,5))

                # identify if var_ is categorical or continuous
                if df[var_].dtype == 'object':
                    # if categorical, use boxplot
                    df.boxplot(column=feature,by=var_,ax=axes)

                else:
                    # if continuous, use scatter
                    df.plot.scatter(x=var_,y=feature,ax=axes)
                plt.xticks(rotation=45)
                fig.suptitle(feature)
                fig.savefig(os.path.join(vizout,f'{feature}.png'))
                plt.close('all')
            except Exception as e:
                save_dict_to_json(os.path.join(vizout,f'{feature}.err'),{'error':str(e)})
                print(f'Error in {feature} {e}')
                print(traceback.format_exc())
                pass



    # Complete collapse (no axes)

    for var_ in ['dataset','age','sex','group']:

        vizout= os.path.join(OUTPUTBASE,f'{var_}_feature_collapsed')
        os.makedirs(vizout,exist_ok=True)

        for feature in df_long['feature'].unique():
            thisdf= df_long[df_long['feature']==feature]
            if 'feature' in feature:# or feature in ['age','group']:
                try:
                    fig,axes= plt.subplots(1,1,figsize=(15,5))

                    # identify if var_ is categorical or continuous
                    if thisdf[var_].dtype == 'object':
                        # if categorical, use boxplot
                        thisdf.boxplot(column='value',by=var_,ax=axes)
                    else:
                        # if continuous, use scatter
                        thisdf.plot.scatter(x=var_,y='value',ax=axes)

                    fig.suptitle(feature)
                    plt.xticks(rotation=45)
                    fig.savefig(os.path.join(vizout,f'{feature}.png'))
                    plt.close('all')
                except Exception as e:
                    save_dict_to_json(os.path.join(vizout,f'{feature}.err'),{'feature':feature,'error':str(e)})
                    print(f'Error in {feature} {e}')
                    print(traceback.format_exc())
                    pass

'''
# Identify feature ontology
feature_ontology = {}

col = df.columns[10]

col

if col.startswith('feature'):
    keyvalpairs = col.split('.')
    getval = lambda x: x.split('-')[1]
    getkey = lambda x: x.split('-')[0]
    feature = getval(keyvalpairs[0])
    if not feature in feature_ontology:
        feature_ontology[feature]={}
    
    

for col in df.columns:
    if col.startswith('feature.'):
        feature = col.split('.')[0]
        feature_ontology[feature] = df[col].unique()
'''