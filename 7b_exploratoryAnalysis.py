#use automl environment
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import pandas as pd
import pickle
#import seaborn as sns
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json,save_figs_in_html
import glob
import traceback
import cmasher as cmr
import matplotlib.patches as mpatches

datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
OUTPUT_DIR = PIPELINE['eda']['path']
os.makedirs(OUTPUT_DIR,exist_ok=True)

fold_path = os.path.join(PIPELINE['scalingAndFolding']['path'])
fold_pattern = os.path.join(fold_path,'**','folds-*.pkl')
foldinstances = glob.glob(fold_pattern,recursive=True)
#foldinstances = [x for x in foldinstances if 'folding' in x]
scalingAndFolding_path = PIPELINE['scalingAndFolding']['path']
eda_path = PIPELINE['eda']['path']
for _foldpath in foldinstances:
    foldingtype=os.path.splitext(os.path.basename(_foldpath))[0]
    foldcomb = os.path.basename(_foldpath).split('-')[1].split('.')[0]
    foldsavepath = _foldpath.replace(scalingAndFolding_path, eda_path)
    os.makedirs(foldsavepath,exist_ok=True)


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
    except:
        raise Exception(f'Error in {_foldpath}:\n{traceback.format_exc()}')



    dfstats=df.copy()
    dfXstats=dfX.copy()
    # rename the columns with dots, needed for statsmodels' ols

    #TODO: maybe make this transform configurable to make plot ready names for features
    dfstats.columns = dfstats.columns.str.replace('.', '_')
    dfXstats.columns = dfXstats.columns.str.replace('.', '_')

    category_vars = PIPELINE['eda']['category_vars']

    # List of features to analyze
    features_of_interest = dfXstats.columns.tolist()

    for cat in category_vars:
        # Convert 'center' and 'gender' to categorical variables
        dfstats[cat] = dfstats[cat].astype('category')


    # Create a DataFrame to store the results
    results_df = []#pd.DataFrame(columns=['Feature', 'Group', 'F-value', 'p-value', 'Corrected p-value', 'Significant'])

    # Define your desired rejection thresholdx;'
    desired_threshold = PIPELINE['eda']['desired_threshold'] # alpha for significance after multiple test correction

    # Loop through each feature
    for feature in features_of_interest:
        formula = PIPELINE['eda']['formula'].replace('{feature}',feature)
        # but Treatment(reference='hc') will use healthy subjects as reference in case of multiple diagnosis
        
        # Fit the model with robust standard errors to account for heteroscedasticity
        model = ols(formula, data=dfstats).fit()

        # Perform ANCOVA
        anova_table = sm.stats.anova_lm(model, typ=2)


        # Extract F-value and p-value
        f_value = anova_table['F'].iloc[0]
        p_value = anova_table['PR(>F)'].iloc[0]

        # Correct for multiple testing using FDR correction
        reject, corrected_p_value, _, _ = multipletests([p_value], method='fdr_by')

        # Check if the corrected p-value is below the desired threshold
        if corrected_p_value[0] < desired_threshold:
            # Append results to the DataFrame
            results_df.append({
                'Feature': feature,
                'Group': formula.split('~')[1].strip(),
                'F-value': f_value,
                'p-value': p_value,
                'Corrected p-value': corrected_p_value[0],
                'Significant': reject[0]
            })#

    os.makedirs(foldsavepath,exist_ok=True)

    results_df = pd.DataFrame.from_dict(results_df)#, ignore_index=True)
    results_df.to_csv(os.path.join(foldsavepath,f'{foldcomb}-resultsols.csv'))
    # Filter and sort significant results by F-value
    significant_results = results_df[results_df['Significant']].sort_values(by='F-value', ascending=False)
    significant_results.to_csv(os.path.join(foldsavepath,f'{foldcomb}-resultsols@significant.csv'))

    #results_df.to_csv()
    sns.set_style('white')
    sns.set(font="Consolas")

    # Create a vertical barplot for significant features with a wider figure
    plt.figure(figsize=(15, 6))  # Adjust the width (12) and height (6) as needed

    plt.bar(significant_results['Feature'], significant_results['F-value'])
    plt.xlabel('Feature')
    plt.ylabel('F-value')
    plt.title(f'{foldingtype} - Significant Features after FDR - BY correction (q-val < {desired_threshold})')
    plt.xticks(rotation=45, ha='right', size=10)
    plt.tight_layout()  # Ensures labels and titles are not cut off
    plt.savefig(os.path.join(foldsavepath,f'{foldcomb}-resultsols@significant.png'))
    plt.close('all')
    # Normalize F-values to range [0, 1]
    normalized_f_values = (significant_results['F-value'] - significant_results['F-value'].min()) / (
            significant_results['F-value'].max() - significant_results['F-value'].min())

    # Use Magma color palette for the barplot based on normalized F-values
    cmap = cmr.gem_r

    # Create a vertical barplot for significant features with a wider figure
    plt.figure(figsize=(15, 6))

    # Plotting with color based on normalized F-values
    plt.bar(significant_results['Feature'], significant_results['F-value'], color=cmap(normalized_f_values))
    plt.xlabel('Feature')
    plt.ylabel('F-value')
    plt.title(f'{foldingtype} - Significant Features after FDR - BY correction (q-val < {desired_threshold})')
    plt.xticks(rotation=45, ha='right', size=10)
    plt.tight_layout()
    plt.savefig(os.path.join(foldsavepath,f'{foldcomb}-resultsols@significant@sns.png'))
    plt.close('all')

    ## Color by categorical_mapping

    for catmap,catmapdict in PIPELINE['eda']['feature_categories'].items() :
        category_mapping = lambda x:x
        exec(catmapdict['category_mapping']) # category_mapping defined here
        
        # Create a new column "type_features" based on conditions
        category_map_name = catmapdict['category_name']
        input_column_mapping = catmapdict['input_column_mapping']
        significant_results[category_map_name]=significant_results[input_column_mapping].apply(category_mapping)
        significant_results[category_map_name].unique()
        # Sort F-values within each subplot in descending order
        significant_results_sorted = significant_results.sort_values(by=[category_map_name, 'F-value'], ascending=[True, False])

        # Set up colors for different types
        # type_colors = {'spectral': 'red', 'complexity': 'blue', 'connectivity': 'green' }
        type_colors = {x:cmr.gem_r(i/len(significant_results[category_map_name].unique())) for i,x in enumerate(significant_results[category_map_name].unique())}


        height_ratios = []
        for i, (type_feature, color) in enumerate(type_colors.items()):
            subset = significant_results_sorted[significant_results_sorted[category_map_name] == type_feature]
            height_ratios.append(len(subset)/len(significant_results_sorted))

        # Create a figure and axis with different heights for the subplots
        fig, axes = plt.subplots(nrows=len(type_colors), ncols=1, figsize=(8, 4 * len(type_colors)),
                                sharex=True, gridspec_kw={'height_ratios': height_ratios})

        # Loop through types and create horizontal bar plots with different colors
        for i, (type_feature, color) in enumerate(type_colors.items()):
            subset = significant_results_sorted[significant_results_sorted[category_map_name] == type_feature]


            axes[i].barh(subset[input_column_mapping], subset['F-value'], color=color)
            axes[i].set_xlabel('F-value')
            axes[i].set_title(f'{type_feature.capitalize()} Features - F-values')
            axes[i].tick_params(axis='y', labelrotation=0, size=8)  # Set rotation to 0 for horizontal labels on y-axis
            axes[i].set_ylabel(input_column_mapping)

            # Adjust subplot size for connectivity
            if type_feature == 'connectivity':
                pass
            else:
                axes[i].invert_yaxis()  # Invert y-axis to have higher F-values at the top

        # Adjust layout for better appearance
        plt.savefig(os.path.join(foldsavepath,f'{foldcomb}-resultsols@significant@{category_map_name}.png'))
        plt.close('all')
    break
