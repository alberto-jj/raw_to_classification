import os
import shutil
import yaml
import json

from sovabids.parsers import placeholder_to_regex,_modify_entities_of_placeholder_pattern
from sovabids.rules import apply_rules,load_rules
from sovabids.dicts import deep_merge_N
from sovabids.datasets import make_dummy_dataset,save_dummy_vhdr,save_dummy_cnt
from sovabids.convert import convert_them

DEF_DATASET_PARAMS ={'PATTERN':'T%task%/S%session%/sub%subject%_%acquisition%_%run%',
'DATASET' : 'DUMMY',
'NSUBS' : 2,
'NTASKS' : 2,
'NRUNS' : 1,
'NSESSIONS' : 1,
'NACQS' : 1,
}


def dummy_dataset(pattern_type='placeholder',mode='python',format='.vhdr', data_params = DEF_DATASET_PARAMS):

    # Getting current file path and then going to _data directory
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir,'..','data')
    data_dir = os.path.abspath(data_dir)

    # Defining relevant conversion paths
    dataset_name = data_params.get('DATASET','DUMMY')
    test_root = os.path.join(data_dir,dataset_name)
    input_root = os.path.join(test_root,dataset_name+'_SOURCE')
    mode_str = '_' + mode
    bids_path = os.path.join(test_root,dataset_name+'_BIDS'+'_'+pattern_type+mode_str+'_'+format.replace('.',''))

    # Make example File
    if format == '.vhdr':
        example_fpath = save_dummy_vhdr(os.path.join(data_dir,'dummy.vhdr'))
    elif format == '.cnt':
        example_fpath = save_dummy_cnt(os.path.join(data_dir,'dummy.cnt'))

    # PARAMS for making the dummy dataset
    DATA_PARAMS ={ 'EXAMPLE':example_fpath,
        'ROOT' : input_root
    }
    DATA_PARAMS.update(data_params)

    # Preparing directories
    dirs = [input_root,bids_path] #dont include test_root for saving multiple conversions
    for dir in dirs:
        try:
            shutil.rmtree(dir)
        except:
            pass

    [os.makedirs(dir,exist_ok=True) for dir in dirs]

    # Generating the dummy dataset
    make_dummy_dataset(**DATA_PARAMS)

    # Making rules for the dummy conversion

    # Gotta fix the pattern that wrote the dataset to the notation of the rules file
    FIXED_PATTERN =DATA_PARAMS.get('PATTERN',None)

    FIXED_PATTERN = _modify_entities_of_placeholder_pattern(FIXED_PATTERN,'append')
    FIXED_PATTERN = FIXED_PATTERN + format

    # Making the rules dictionary
    data={
    'dataset_description':
        {
            'Name':'Dummy',
            'Authors':['A1','A2'],
        },
    'sidecar':  
        {
            'PowerLineFrequency' : 50,
            'EEGReference':'FCz',
            'SoftwareFilters':{"Anti-aliasing filter": {"half-amplitude cutoff (Hz)": 500, "Roll-off": "6dB/Octave"}}
        },
    'non-bids':
        {
        'eeg_extension':format,
        'path_analysis':{'pattern':FIXED_PATTERN},
        'code_execution':['print(\'some good code\')','print(raw.info)','print(some bad code)']
        },
    'channels':
        {'name':{'1':'ECG_CHAN','2':'EOG_CHAN'}, #Note example vhdr and CNT have these channels
        'type':{'ECG_CHAN':'ECG','EOG_CHAN':'EOG'}} # Names (keys) are after the rename of the previous line
    }

    if pattern_type == 'regex':
        FIXED_PATTERN_RE,fields = placeholder_to_regex(FIXED_PATTERN)
        dregex = {'non-bids':{'path_analysis':{'fields':fields,'pattern':FIXED_PATTERN_RE}}}
        data = deep_merge_N([data,dregex])
    # Writing the rules file
    outputname = dataset_name+'_rules'+'_'+pattern_type+'.yml'

    full_rules_path = os.path.join(test_root,outputname)
    with open(full_rules_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
def test_dummy_dataset():

    DATA_PARAMS_1 = DEF_DATASET_PARAMS.copy()
    DATA_PARAMS_1['DATASET'] = 'DUMMY1'

    # apparently it cannot download the cnt consistenly on the github actions machine
    #dummy_dataset('placeholder',format='.cnt') # Test cnt conversion
    dummy_dataset('placeholder', format='.vhdr', data_params=DATA_PARAMS_1)  # Test vhdr conversion

    DATA_PARAMS_2 = DEF_DATASET_PARAMS.copy()
    DATA_PARAMS_2['DATASET'] = 'DUMMY2'

    dummy_dataset('placeholder', format='.vhdr', data_params=DATA_PARAMS_2)

    print("Dummy dataset test completed successfully.")

def test_inspect():
    """
    This function is a placeholder for the inspect function.
    It should be implemented to inspect the dummy datasets created above.
    """

    from eeg_raw_to_classification.pipelines.inspect import pipeline_inspect
    pipeline_inspect("./project_files/dummy_pipeline.yml")
    print("Inspect test completed successfully.")

def test_dataset2bids():
    """
    This function is a placeholder for the dataset2bids function.
    It should be implemented to convert the dummy datasets to BIDS format.
    """
    from eeg_raw_to_classification.pipelines.dataset2bids import pipeline_dataset2bids
    pipeline_dataset2bids("./project_files/dummy_pipeline.yml")
    print("Dataset to BIDS test completed successfully.")


def test_participants():
    """
    This function is a placeholder for the participants function.
    It should be implemented to create a participants.tsv file for the dummy datasets.
    """
    from eeg_raw_to_classification.pipelines.participants import pipeline_participants
    pipeline_participants("./project_files/dummy_pipeline.yml")
    print("Participants test completed successfully.")

def test_preprocess():
    """
    This function is a placeholder for the preprocess function.
    It should be implemented to preprocess the dummy datasets.
    """
    from eeg_raw_to_classification.pipelines.preprocessing import pipeline_preprocess
    pipeline_yml = "./project_files/dummy_pipeline.yml"
    max_files = None
    external_jobs = 1
    internal_jobs = 1
    retry_errors = True
    raise_on_error = True
    index = None
    only_total = True

    pipeline_preprocess(pipeline_yml, max_files=max_files, external_njobs=external_jobs,
                        internal_njobs=internal_jobs, retry_errors=retry_errors,
                        raise_on_error=raise_on_error, index=index, only_total=only_total)

    only_total = False
    pipeline_preprocess(pipeline_yml, max_files=max_files, external_njobs=external_jobs,
                        internal_njobs=internal_jobs, retry_errors=retry_errors,
                        raise_on_error=raise_on_error, index=index, only_total=only_total)
    print("Preprocess test completed successfully.")

def test_prep_inspect():
    """
    This function is a placeholder for the prep_inspect function.
    It should be implemented to inspect the preprocessing of the dummy datasets.
    """
    from eeg_raw_to_classification.pipelines.inspect_prep import pipeline_inspect_prep
    pipeline_inspect_prep("./project_files/dummy_pipeline.yml")
    print("Prep inspect test completed successfully.")

def test_features():
    from eeg_raw_to_classification.pipelines.features import pipeline_features
    pipeline_features("./project_files/dummy_pipeline.yml")
    print("Features test completed successfully.")
if __name__ == '__main__':
    # test_dummy_dataset()
    test_inspect()
    # test_dataset2bids()
    # test_participants()
    # test_preprocess()
    # test_prep_inspect()
    #test_features()


    print('ok')