from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json,save_figs_in_html
import os
import pandas as pd

datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
OUTPUT_DIR = PIPELINE['ml']['path']
os.makedirs(OUTPUT_DIR,exist_ok=True)

df_path = os.path.join(PIPELINE['aggregate']['path'],PIPELINE['aggregate']['filename'])
df = pd.read_csv(df_path)

# Filter Examples
df = df.copy()[df['group']=='HC']


targets=['age']
drops_harmonization=['id','group','subject','task']
stratify_var = 'sex'
covariables =['age','sex','dataset']
all_drops=drops_harmonization+targets




from sklearn.model_selection import train_test_split
from reComBat import reComBat

# Previous transformation and filters

# Harmonization Feature Generator Class

from autogluon.features.generators import AbstractFeatureGenerator


# Feature generator to add k to all values of integer features.
class reCombatFeatureGenerator(AbstractFeatureGenerator):
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        # Here we can specify any logic we want to make a stateful feature generator based on the data.
        # Just call _transform since this isn't a stateful feature generator.
        X_out = self._transform(X)
        # return the output and the new special types of the data. For this generator, we don't add any new special types, so just return the input special types
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        # Here we can specify the logic taken to convert input data to output data post-fit. Here we can reference any variables created during fit if the generator is stateful.
        # Because this feature generator is not stateful, we simply add k to all features.
        return X + self.k

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_INT])  # This limits input features to only integers. We can assume that the input to _fit_transform and _transform only contain the data post-applying this filter.


################################
# Fit custom feature generator #
################################

plus_three_feature_generator = PlusKFeatureGenerator(k=3, verbosity=3)
X_transform = plus_three_feature_generator.fit_transform(X=X)
print(X_transform.head(5))

#https://github.com/Imaging-AI-for-Health-virtual-lab/harmonizer/blob/main/Harmonizer.py
#https://github.com/autogluon/autogluon/blob/f269d65a0df646f1b58bb14bfa583c140581ed06/core/src/autogluon/core/models/abstract/abstract_model.py#L405-L419
#https://github.com/autogluon/autogluon/issues/4116
#https://github.com/autogluon/autogluon/blob/master/examples/tabular/example_custom_feature_generator.py

# Splitting the data into train and test datasets
# test: out of sample validation (val) holdout data
# train: what will be used in training (train and test to be split into folds later)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :], df, test_size=0.3, random_state=1997, stratify=df["group"])

model = reComBat(parametric=True,
                 model='elastic_net',
                 config={'alpha': 1e-5},
                 n_jobs=7,
                 verbose=True)

# Creating a subset of X_train and renaming the column
covars = X_train.iloc[:, 1:5].copy()
covars.rename(columns={'age': 'age_numerical'}, inplace=True)
covars.center = covars.center.astype(str)
covars.group = covars.group.astype(str)
covars.gender = covars.gender.astype(str)


# Fitting the reComBat model on the train data
model.fit(data=X_train.iloc[:,5:], batches=covars.center,X=covars.drop(['center'],axis=1))

# Transforming the train data using the fitted model
transformed_train_data = model.transform(data=X_train.iloc[:,5:], batches=covars.center,X=covars.drop(['center'],axis=1))

# Merging covariates and transformed train data
X_train_harmonized = pd.concat([X_train.iloc[:,:5], transformed_train_data], axis=1)
train_combat = X_train_harmonized.copy()
train_combat['split'] = 'train'

# Creating a subset of X_train and renaming the column
covars = X_test.iloc[:, 1:5].copy()
covars.rename(columns={'age': 'age_numerical'}, inplace=True)
covars.center = covars.center.astype(str)
covars.group = covars.group.astype(str)
covars.gender = covars.gender.astype(str)

# Transforming the test data using the fitted model
transformed_test_data = model.transform(data=X_test.iloc[:,5:], batches=covars.center,X=covars.drop(['center'],axis=1))

# Merging covariates and transformed train data
X_test_harmonized = pd.concat([X_test.iloc[:,:5], transformed_test_data], axis=1)
test_combat = X_test_harmonized.copy()
test_combat['split'] = 'test'


all_combat_df = pd.concat([train_combat, test_combat], axis=0)
all_combat_df