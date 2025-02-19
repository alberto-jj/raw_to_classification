#use automl environment
import pandas as pd
import pickle
from itertools import product
#import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
import copy
import matplotlib.pyplot as plt
#import umap
#import umap.plot
import os
import yaml
import json
from featurewiz import FeatureWiz
from sklearn.model_selection import train_test_split
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json,save_figs_in_html
import itertools
from reComBat import reComBat
from sklearn.decomposition import PCA



PIPELINE = load_yaml(f'pipeline.yml')
datasets = load_yaml(PIPELINE['datasets_file'])
PROJECT = PIPELINE['project']
OUTPUT_DIR_BASE = PIPELINE['scalingAndFolding']['path'].replace('%PROJECT%,PROJECT')

for aggregate_folder in PIPELINE['scalingAndFolding']['aggregate_folders']:
    CFG = PIPELINE['scalingAndFolding']
    OUTPUT_DIR=os.path.join(OUTPUT_DIR_BASE,aggregate_folder)
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    input_desc='@raw'
    df_path = os.path.join(PIPELINE['aggregate']['path'],aggregate_folder,PIPELINE['aggregate']['filename']+input_desc+'.csv')
    df = pd.read_csv(df_path)

    #########################
    # apply filtering of data points here
    #########################

    csvfilename=os.path.splitext(os.path.basename(df_path))[0]
    desc=csvfilename.split('@')[1].replace('.csv','')
    csvfilename=csvfilename.replace('@'+desc,'@DESC')
    csvfolder=os.path.dirname(df_path)
    csvpath=os.path.join(csvfolder,csvfilename)+'.csv'
    df = df.dropna()
    df.to_csv(csvpath.replace('@DESC','@noNaNs'),index=False)

    # keep only healthy
    df['group'].unique()
    df['group'].value_counts()
    df_selected = df[df['group']=='HC']


    ## final save
    df_final = df_selected.copy()
    df_final.to_csv(csvpath.replace('@DESC','@final'),index=False)
    

