# use automl environment
import argparse
import pandas as pd
import pickle
from itertools import product
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from eeg_raw_to_classification.utils import load_yaml
import itertools

def main(pipeline_file):
    PIPELINE = load_yaml(pipeline_file)

    datasets = load_yaml(PIPELINE['datasets_file'])
    PROJECT = PIPELINE['project']
    OUTPUT_DIR_BASE = PIPELINE['scalingAndFolding']['path'].replace('%PROJECT%', PROJECT)

    for aggregate_folder in PIPELINE['scalingAndFolding']['aggregate_folders']:
        CFG = PIPELINE['scalingAndFolding']
        OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, aggregate_folder)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        input_desc = '@raw'
        df_path = os.path.join(PIPELINE['aggregate']['path'].replace('%PROJECT%', PROJECT), aggregate_folder, PIPELINE['aggregate']['filename'] + input_desc + '.csv')
        df = pd.read_csv(df_path)

        #########################
        # apply filtering of data points here
        # change the code below to filter the data points
        #########################

        csvfilename = os.path.splitext(os.path.basename(df_path))[0]
        desc = csvfilename.split('@')[1].replace('.csv', '')
        csvfilename = csvfilename.replace('@' + desc, '@DESC')
        csvfolder = os.path.dirname(df_path)
        csvpath = os.path.join(csvfolder, csvfilename) + '.csv'
        df = df.dropna()
        df.to_csv(csvpath.replace('@DESC', '@noNaNs'), index=False)

        # keep only healthy
        df['group'].unique()
        df['group'].value_counts()
        df_selected = df[df['group'] == 'HC']

        ## final save
        df_final = df_selected.copy()
        df_final.to_csv(csvpath.replace('@DESC', '@final'), index=False)
        print(f'Saved {csvpath.replace("@DESC", "@final")}, Original shape: {df.shape}, Final shape: {df_final.shape}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pipeline.yml file.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline.yml file')
    args = parser.parse_args()
    main(args.pipeline_file)
