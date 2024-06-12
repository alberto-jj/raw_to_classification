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

datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
OUTPUT_DIR_BASE = PIPELINE['scalingAndFolding']['path']

for aggregate_folder in PIPELINE['scalingAndFolding']['aggregate_folders']:
    CFG = PIPELINE['scalingAndFolding']
    OUTPUT_DIR=os.path.join(OUTPUT_DIR_BASE,aggregate_folder)
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    df_path = os.path.join(PIPELINE['aggregate']['path'],aggregate_folder,PIPELINE['aggregate']['filename'])
    df = pd.read_csv(df_path)

    # SETUP
    RANDOM_STATE = CFG['random_state']
    targets=CFG['targets']
    dropsToOnlyFeatures= CFG['dropsToOnlyFeatures'] + targets
    stratifiedvars= CFG['stratifiedvars'] # these are categorical in general... , they should be the same as the covars in combat?
    # you may include or not some variables if too many combinations dont exist, for example only dataset (site)

    # TODO: we had max_features in the previous harmonization pipeline, should we include it here?
    # MAX_FEATURES=PIPELINE['harmonization']['MAX_FEATURES']
    # if MAX_FEATURES is not None:
    #     df = df.iloc[:,:MAX_FEATURES]

    # TODO: make roi space features here...

    # The thing here is that some combinations may not actually exist in the dataset
    # Guess we should just ignore those and collect up to the lowest represented

    joined_stratifiedvars = [list(x) for i,x in df[stratifiedvars].iterrows()]
    joined_stratifiedvars = ['_'.join(x) for x in joined_stratifiedvars]
    df['combination']=joined_stratifiedvars
    dropsToOnlyFeatures +=['combination']
    existing_combinations = pd.unique(joined_stratifiedvars)
    lowest_representation = pd.value_counts(joined_stratifiedvars).min()

    # Another interesting check... (?)

    for stratifiedvar in stratifiedvars:
        print(df[stratifiedvar].value_counts())

    possibilities = []
    for stratifiedvar in stratifiedvars:
        uni=df[stratifiedvar].unique()
        print(stratifiedvar,list(uni))
        possibilities.append(list(uni))

    # See non existing combinations and report that

    possible_combinations=list(itertools.product(*possibilities))
    possible_combinations=['_'.join(x) for x in possible_combinations]

    nonexisting_combinations = set(possible_combinations).difference(joined_stratifiedvars)

    if len(nonexisting_combinations) > 0:
        print('WARNING, the following combinations dont exist:')
        print(nonexisting_combinations)

    # For now balanced withing the combinations that exist?

    random_seed = 0
    df_balanced = []
    for comb_ in existing_combinations:
        subdf=df[df['combination']==comb_]
        subdf=subdf.sample(n=lowest_representation,replace=False,random_state=RANDOM_STATE)
        df_balanced.append(subdf)

    df_orig = df.copy()

    split_scaling_combs = list(product(CFG['scalings'].keys(),CFG['splits'].keys()))

    for scaling_name,split_name in split_scaling_combs:


        split_cfg = CFG['splits'][split_name]
        scaling_cfg = CFG['scalings'][scaling_name]
        fullname = f'{scaling_name}@{split_name}'
        print(fullname)


        df = pd.concat(df_balanced,ignore_index=True)
        assert lowest_representation*len(existing_combinations) == df.shape[0]
        if 'foldSet' in dropsToOnlyFeatures:
            dropsToOnlyFeatures.remove('foldSet')
        dfX = df.drop(dropsToOnlyFeatures, axis=1)
        dfY = df[targets]
        df['foldSet']=-1
        folds = {}

        save_path = os.path.join(OUTPUT_DIR,f'{scaling_name}_{split_name}')
        os.makedirs(save_path,exist_ok=True)
        if split_cfg['method'] == 'StratifiedKFold':
            NUM_FOLDS = split_cfg['num_folds']
            assert NUM_FOLDS <= lowest_representation

            skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)


            # Verify correct features
            # print(dfX.columns) # maybe this should go to some log...

            # stratification based on the "y" variable in these arguments, hence, combinations
            # But this would work better if all combinations were present...
            # See disparities when printing the value_counts of the stratifiedvars

            for foldi,indexes in enumerate(skf.split(dfX,df['combination'])):
                print(foldi)
                train_index = indexes[0]
                test_index = indexes[1]
                folds[foldi] = {'train':train_index,'test':test_index}

                df['foldSet'].iloc[test_index]=foldi


            assert -1 not in df['foldSet'].unique()


        elif split_cfg['method'] == 'all':
            ## fold -1 will be all_seen data
            folds[-1] = {'train':dfX.index.to_numpy(),'test':dfX.index.to_numpy()}

        elif split_cfg['method'] == 'train_test_split':
            ## fold -1 will be all_seen data
            if split_cfg['stratify']==True:
                stratify = df['combination']
            else:
                stratify = None
            train_index, test_index = train_test_split(dfX.index.to_numpy(), test_size=split_cfg['test_ratio'], random_state=RANDOM_STATE,stratify=stratify)
            folds[-1] = {'train':train_index,'test':test_index}


        # To maintain good account of column transformations
        dropsToOnlyFeatures += ['foldSet']

        # Now with the fold column we should be able to leverage the OneGroupOut of Autogluon
        # But first apply scaling and transform from the training data of each fold

        for foldi,this_fold in folds.items():
            train_index = this_fold['train']
            test_index = this_fold['test']

            df_train = df.copy().iloc[train_index]
            df_test = df.copy().iloc[test_index]

            dfX = df.copy().drop(dropsToOnlyFeatures,axis=1)
            dfX_train = dfX.copy().iloc[train_index]
            dfX_test = dfX.copy().iloc[test_index]

            dfY_train = dfY.copy().iloc[train_index]
            dfY_test = dfY.copy().iloc[test_index]

            folds[foldi]['counts_train'] = {f'{var}_train':df_train[var].value_counts() for var in stratifiedvars}
            folds[foldi]['counts_test'] = {f'{var}_test':df_test[var].value_counts() for var in stratifiedvars}

            # Grab Unseen data (test sets across all folds)
            dfXacross_test = []
            dfYacross_test = []
            dfacross = []

            # we will have to save this scaled data for each fold as multiple scaling exist for the same row depending on which fold it is
            # remember that we scale on the training indexes (not unique across folds) rather than the test indexes (unique)

            if 'reCombat' in scaling_cfg['method']:
                # COMBAT SETUP

                combat_covars = scaling_cfg['covars']
                combat_rename = scaling_cfg['rename']
                combat_categorical = scaling_cfg['categorical']
                combat_batch = scaling_cfg['batch'] # which should be the one we drop from covars for the design matrix X in combat
                combat_init = scaling_cfg['init']

                combat_model = reComBat(**combat_init)

                #Defining adjustment covariates from X_train to perform my desired re-scaling
                covars_train = df_train[combat_covars].copy()
                covars_train = covars_train.rename(columns=combat_rename, inplace=False)
                for col in combat_categorical:
                    covars_train[col] = covars_train[col].astype(str)

                #Defining adjustment covariates from X_train to perform my desired re-scaling
                covars_test = df_test[combat_covars].copy()
                covars_test = covars_test.rename(columns=combat_rename, inplace=False)
                for col in combat_categorical:
                    covars_test[col] = covars_test[col].astype(str)


                #Fitting the re-scaling (reComBat) model on the train data
                combat_model.fit(data=dfX_train, batches=covars_train[combat_batch],X=covars_train.drop([combat_batch],axis=1))

                #Re-scaling the train data using the fitted model
                dfX_train = combat_model.transform(data=dfX_train.copy(), batches=covars_train[combat_batch],X=covars_train.drop([combat_batch],axis=1))
                dfX_test = combat_model.transform(data=dfX_test.copy(), batches=covars_test[combat_batch],X=covars_test.drop([combat_batch],axis=1))

            elif 'StandardScaler' in scaling_cfg['method']:
                scaler = StandardScaler()
                scaler = scaler.fit(dfX_train)
                dfX_train = pd.DataFrame(scaler.transform(dfX_train.copy()), columns=dfX_train.columns)
                dfX_test = pd.DataFrame(scaler.transform(dfX_test.copy()), columns=dfX_test.columns)

            else:
                # NO SCALING
                pass
            folds[foldi]['X_train'] = dfX_train.copy()
            folds[foldi]['X_test'] = dfX_test.copy() # this would be unique, but for simplicity just maintain with the same structure we have
            folds[foldi]['Y_train'] = dfY_train.copy()
            folds[foldi]['Y_test'] = dfY_test.copy()
            folds[foldi]['df_train'] = df_train.copy()
            folds[foldi]['df_train']['foldIter']=foldi
            folds[foldi]['df_test'] = df_test.copy()
            folds[foldi]['df_test']['foldIter']=foldi
        # Save the folds in pkl
        with open(os.path.join(save_path,f'folds-{fullname}.pkl'),'wb') as f:
            pickle.dump(folds,f)

        for foldi,this_fold in folds.items():
            dfXacross_test.append(this_fold['X_test'])
            dfYacross_test.append(this_fold['Y_test'])
            dfacross.append(this_fold['df_test'])
        dfXacross_test = pd.concat(dfXacross_test,axis=0,ignore_index=True)
        dfYacross_test = pd.concat(dfYacross_test,axis=0,ignore_index=True)
        dfacross = pd.concat(dfacross,axis=0,ignore_index=True)
        
        dfXBefore =df.copy().drop(dropsToOnlyFeatures,axis=1) # The before data is the complete feature set, the folds dont matter here
        dfBefore = df.copy()
        dfYBefore = df.copy()[targets]

        # Scaling Effect

        scaling_effect={
            'after(unseen-in-all-folds)':{
                'X':dfXacross_test,
                'Y':dfYacross_test,
                'df':dfacross
            },
            'before(no-folds)':{
                'X':dfXBefore,
                'Y':dfYBefore,
                'df':dfBefore
            }
        }

        with open(os.path.join(save_path,'scaling_effect.pkl'),'wb') as f:
            pickle.dump(scaling_effect,f)

        # visualization

        for vizname,viz_cfg in CFG['visualizations'].items():

            # Note that When split_method is all, the unseen data is the same as the seen data
            # so that would be the one to review for seeing what happens when the scaler knows all the data
            
            for effect,_scaling_effect in scaling_effect.items():
            
                _dfXacross_test = _scaling_effect['X']
                _dfYacross_test = _scaling_effect['Y']
                _dfacross = _scaling_effect['df']
                phase = effect

                # TODO: waterfall plot from the code we did?

                if viz_cfg['method'] == 'PCA':

                    # TODO: save data to produce customized plots for paper later

                    # TODO: Look at Alberto's plots for inspiration
                    # PCA
                    pca = PCA(n_components=viz_cfg['n_components'])
                    pca.fit(_dfXacross_test)
                    dfXacross_test_pca = pca.transform(_dfXacross_test)
                    dfXacross_test_pca = pd.DataFrame(dfXacross_test_pca,columns=['PC1','PC2'])

                    # Plot
                    fig, ax = plt.subplots()
                    color_by = viz_cfg['color_by']
                    color_by = _dfacross[color_by]


                    # Get unique categories
                    categories = color_by.unique()

                    # Create a colormap and get colors for each category
                    cmap = plt.get_cmap(viz_cfg['colormap'], len(categories))
                    colors = [cmap(i) for i in range(len(categories))]

                    # Map categories to colors
                    color_map = dict(zip(categories, colors))
                    c = [color_map[category] for category in color_by]
                    scatter = ax.scatter(dfXacross_test_pca['PC1'],dfXacross_test_pca['PC2'],c=c)
                    ax.legend(*scatter.legend_elements(), title=viz_cfg['color_by'])
                    ax.set_title('PCA')

                    # Create a proxy artist for each category and add it to the legend
                    legend_elements = [plt.scatter([], [], color=color, marker='o', label=category)
                                    for category, color in color_map.items()]

                    ax.legend(handles=legend_elements, title=viz_cfg['color_by'])
                    plt.savefig(os.path.join(save_path,f'PCA_{phase}.png'))

                    with open(os.path.join(save_path,'scaling_effectData.pkl'),'wb') as f:
                        pickle.dump({'dfXacross_test_pca':dfXacross_test_pca},f)

        # Feature Engineering with all unseen transformed data
        # when split_method is all, the unseen data is the same as the seen data, so that would be the best to use for feature engineering insight
        # (the scaler knows all the data)

        for effect,_scaling_effect in scaling_effect.items():
            fwiz = FeatureWiz(**CFG['featurewiz']['init'])
            print(fullname,effect)
            print(_scaling_effect['X'].shape,_scaling_effect['Y'].shape)
            X_train_selected = fwiz.fit(_scaling_effect['X'], _scaling_effect['Y'])

            save_dict_to_json(os.path.join(save_path,f'fwiz_{effect}.json'),{'features':fwiz.features})
            with open(os.path.join(save_path,f'fwiz_{effect}.pkl'),'wb') as f:
                pickle.dump(fwiz,f)

    ## BUILD A REDUNDANT DATAFRAME WITH TEST AND TRAIN OF EACH FOLD IN SEPARATE ROWS, WITH EXTRA COLUMNS OF FOLDS AND TYPE (TRAIN OR TEST)
    ## FOR AUTOMLJAR PASS THE APPROPIATE INDEXES FOR EACH FOLD'S TRAIN AND TEST
    ## FOR AUTOGULON IS HARDER, WE HAVE TO MANIPULATE THE LEAVEONEGROUPOUT
    ## WHEN TRAINING FOLD 1 the training will include folds 2,3,4,5 training data which include for example data from fold 2 with multiple scalings
    ## SO THIS IS NOT TRIVIAL...
    ## MAYBE THE BEST IS TO USE AUTOGLUON FROM THE OUTSIDE, THAT IS PASS ONE FOR PER AUTOGLUON CALL (NO BAGGING/FOLDS)
    ## AND COLLECT THE RESULTING DATAFRAMES FROM STATITSTICS

