import os
import scipy.io as sio
import mne
import numpy as np
import glob
import scipy
from mne.io import read_raw
from mne import read_epochs
import pandas as pd
from sovabids.parsers import parse_from_placeholder
from mne_bids import BIDSPath, write_raw_bids
import traceback
import pdb
from pprint import pprint
def thanks_jordan_venkatesh(source_path, bids_path, DATASET_CFG, pipeline_cfg):

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  # Adjust width as needed


    MOUNT = pipeline_cfg.get('mount', None)

    ID_file = DATASET_CFG.get('bidsify', {}).get('ID_file', {}).get(MOUNT, None)

    df_ids = pd.read_excel(ID_file, header=2)

    # drop nans
    df_ids = df_ids.dropna(subset=['ID'])

    # pass _ to -

    df_ids['ID'] = df_ids['ID'].str.replace('_', '-', regex=False)

    rename = {
        'Unnamed: 0':'Unnamed',
        'ID':'ID',
        'subject':'number',
        'Unnamed: 3':'label',
    }

    df_ids = df_ids.rename(columns=rename)

    SOURCE_PATH = source_path

    folders = glob.glob(os.path.join(SOURCE_PATH, '*'), recursive=False)

    # count the number of empty folders

    empty_folders = [folder for folder in folders if os.path.isdir(folder) and not os.listdir(folder)]
    non_empty_folders = [folder for folder in folders if os.path.isdir(folder) and os.listdir(folder)]

    print(f"Number of empty folders: {len(empty_folders)}")
    print(f"Number of non-empty folders: {len(non_empty_folders)}")


    ds_list = glob.glob(os.path.join(SOURCE_PATH, '*.ds'), recursive=False)
    megs = [ds for ds in ds_list if os.path.isdir(ds) and True] # Filter here if needed


    # '/home/yorguin/scratch/data/MEG_LSDV2/meg_data/310714-1_LSD_20140917_Closed1.ds/hz.ds'
    # '/home/yorguin/scratch/data/MEG_LSDV2/meg_data/1_LSD_20140626_01.ds'
    pattern = DATASET_CFG.get('bidsify', {}).get('pattern', {}).get(MOUNT, None)

    megs_parsed = []
    for meg_file in megs:
        try:
            bids_dict = parse_from_placeholder(meg_file,pattern=pattern)
            bids_dict['file']= meg_file  # Add the original file path to the BIDSPath
            #meg = mne.io.read_raw(meg_file, preload=True)  # Load the raw data
            print(f"Parsed BIDS path for {meg_file}: {bids_dict}")
            #print(meg)
            megs_parsed.append(bids_dict)
        except Exception as e:
            print(f"Error parsing BIDS path for {meg_file}: {e}")
    #    break

    # Convert the list of dictionaries to a DataFrame
    df_megs = pd.DataFrame(megs_parsed)

    # for i, row in df_megs.iterrows():
    #     print(row['file'], row['subjectNumber'], row['session'], row['task'])


    # Simplify the path for easier inspection
    def simplify_path(path):
        """Simplify the path by removing the base directory."""
        base_dir = source_path
        return path.replace(base_dir, '')

    df_megs['file'] = df_megs['file'].apply(simplify_path)


    # Drop all MMN tasks
    df_megs = df_megs[~df_megs['task'].str.contains('MMN', na=False)]


    # Correct id as the previous parse_bids gets rid of "-" (non bids compliant)
    def correct_id(id_str):
        if len(id_str) == 7:
            return id_str[:6] + '-' + id_str[6]
        else:
            return id_str

    df_megs['subjectNumber'] = df_megs['subjectNumber'].apply(correct_id)



    ###########################################################################
    # JORDAN NOTES


    id_key = {
                    '1':  '230911-1', 
                    '2':  '010514-1', 
                    '3':  '140514-1', 
                    '4':  '140514-2', 
                    '5':  '050913-1', 
                    '6':  '290514-2', 
                    '7':  '270613-1',
                    # no 8
                    '9':  '260614-1', 
                    '10': '240107-6',
                    '11': '090714-1', 
                    '12': '090714-2',
                    '13': '310714-1',
                    '14': '041213-1', 
                    '15': '070814-1',
                    '16': '070814-2',
                    '17': '200814-1', 
                    '18': '010813-4', 
                    '19': '040914-1', 
                    '20': '040914-2', 
                    }
    pla_key = {
                '1':  ('20140515', "# on disk. 'Closed2' was named simply 'Closed'."),
                '2':  ('20140501', "# on disk"),
                '3':  ('20140528', "# on disk"),
                '4':  ('20140514', "# id was misnamed 280212-2 in ds"),
                '5':  ('20140612', "# on disk"),
                '6':  ('20140529', "# on disk"),
                '7':  ('20140625', "# on disk. 'Opoen1' instead of 'Open1'. Music missing."),
                '9':  ('20140710', "# on disk"),
                '10': ('20140626', "# on disk"),
                '11': ('20140730', "# on disk"),
                '12': ('20140709', "# id was misnamed 070914-2 in ds"),
                '13': ('20140917', "# on disk"),
                '14': ('20140731', "# on disk. 'Music' was saved as 'Audio'"),
                '15': ('20140821', "# on disk. 'Closed1' was named 'Closed'"),
                '16': ('20140807', "# on disk"),
                '17': ('20140903', "# on disk"),
                '18': ('20140820', "# on disk"),
                '19': ('20140918', "# on disk"),
                '20': ('20140904', "# on disk"),
                }
    lsd_key = {
                '1':  ('20140501', "# id was misnamed 231109-1. There is a second music rec named 'Music2'."),
                '2':  ('20140515', "# id was misnamed 300414-1 in ds ('LSD Analysis.xlsx' misreports this mistake)"),
                '3':  ('20140514', "# on disk. There is a second music rec named 'Music2'."),
                '4':  ('20140528', "# on disk"),
                '5':  ('20140529', "# on disk. 'Open1' was saved as 'Open'"),
                '6':  ('20140612', "# on disk"),
                '7':  ('20140611', "# on disk. Music missing. According to Venkatesh, nothing to do here..."),
                '9':  ('20140626', "# on disk"),
                '10': ('20140710', "# on disk"),
                '11': ('20140709', "# on disk"),
                '12': ('20140730', "# on disk"),
                '13': ('20140731', "# on disk"),
                '14': ('20140917', "# on disk"),
                '15': ('20140807', "# on disk. There is a second Video named 'Video2'. Venkatesh says to use 'Video2' instead of Video (Video gives hpi info missing error)"),
                '16': ('20140821', "# on disk"),
                '17': ('20140820', "# on disk"),
                '18': ('20140903', "# id was misnamed 010814-4 in ds"),
                '19': ('20140904', "# on disk (in Supp)"),
                '20': ('20140918', "# on disk (in Supp)"),
                }

    jordan_notes = {}

    for key in id_key.keys():
        jordan_notes[key] = {}
        jordan_notes[key]['id'] = id_key[key]
        jordan_notes[key]['pla'] = pla_key[key]
        jordan_notes[key]['lsd'] = lsd_key[key]

    jordan_df = pd.DataFrame.from_dict(jordan_notes, orient='index')


    ###########################################################################


    # This is an example of a correct subject, there should be Video, Open1, Open2, Closed1, Closed2, Music, MMN1, MMN2, MMN3 tasks (9 tasks) (6 if MMN tasks are not included)
    # 2 sessions: 1 Placebo, 1 LSD (so 9x2 = 18 files) per subject
    correct_tasks = ['Video', 'Open1', 'Open2', 'Closed1', 'Closed2', 'Music'] #, 'MMN1', 'MMN2', 'MMN3']

    def check_subject_tasks(subject_number, df,correct_tasks=correct_tasks,print_df=False):
        good = True
        df2 = df.copy()
        query_df = df2[df2['subjectNumber'] == subject_number].sort_values(by='task', ascending=False)
        #print(query_df)

        for task in query_df['task'].unique():
            if task not in correct_tasks:
                print(f"Subject {subject_number} has an unexpected task: {task}")
                good = False

        for task in correct_tasks:
            if task not in query_df['task'].values:
                print(f"Subject {subject_number} is missing task: {task}")
                good = False
        
        for task in correct_tasks:
            query2_df_task = query_df.copy()[query_df['task'] == task]
            if len(query2_df_task) != 2:
                print(f"Subject {subject_number} has {len(query2_df_task)} files for task {task}, expected 2.")
                good = False

        if good:
            print(f"Subject {subject_number} has exactly expected tasks and files.")
        else:
            print(f"Subject {subject_number} has issues with tasks.")

        if print_df:
            print(query_df)
        return good,query_df


    ##### Example of a correct subject ########
    subject = '200814-1'
    check_subject_tasks(subject,df_megs)

    #####################


    def check_all_subjects(df, correct_tasks=correct_tasks):
        df_ = df.copy()
        bad_subjects = []
        good_subjects = []
        for subject in df_['subjectNumber'].unique().tolist():
            print(f"Checking subject {subject}")
            good,query_df = check_subject_tasks(subject, df_.copy(),correct_tasks=correct_tasks, print_df=False)
            if not good:
                bad_subjects.append(subject)
            else:
                print(f"Subject {subject} is good.")
                good_subjects.append(subject)
            print('#'*50)
        return bad_subjects, good_subjects

    check_all_subjects(df_megs, correct_tasks=correct_tasks)
    #####################################################################

    # Drop unexpected subjects : TEST, 1,2.

    unexpected_subjects = ['TEST', '1', '2', '23041451']

    unused_subjects_for_some_reason = ['140614-1', '300114-1', '071012-1']
    unexpected_subjects.extend(unused_subjects_for_some_reason)

    df_megs = df_megs[~df_megs['subjectNumber'].isin(unexpected_subjects)]

    ####################################################################
    # Make corrections based on Jordan's notes

    df_megs =  df_megs.sort_values(by=['subjectNumber'], ascending=True)
    df_megs_corrected = df_megs.copy()

    ###################################################################

    def placebo_lsd_logic(df,jordan_df, this_subject):
        df_megs_corrected = df.copy()
        # Now apply session placebo/lsd logic

        placebo_session = jordan_df[jordan_df['id'] == this_subject]['pla'].values[0][0]
        lsd_session = jordan_df[jordan_df['id'] == this_subject]['lsd'].values[0][0]

        df_megs_corrected.loc[(df_megs_corrected['subjectNumber'] == this_subject) & (df_megs_corrected['session'] == placebo_session), 'session'] = 'placebo'
        df_megs_corrected.loc[(df_megs_corrected['subjectNumber'] == this_subject) & (df_megs_corrected['session'] == lsd_session), 'session'] = 'lsd'

        print(df_megs_corrected[df_megs_corrected['subjectNumber'] == this_subject])

        check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)
        return df_megs_corrected

    def print_jordan_notes(this_subject):
        notes_row = jordan_df[jordan_df['id'] == this_subject].iloc[0]
        for col in notes_row.index:
            print(f"{col}: {notes_row[col]}")

    ####################################################################

    # Apply placebo/lsd logic to initial good megs

    initial_bads,initial_goods = check_all_subjects(df_megs, correct_tasks=correct_tasks)

    for this_subject in initial_goods:
        print(f"Applying placebo/lsd logic to subject {this_subject}")
        df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    ####################################################################

    # 070814-1
    this_subject = '070814-1'
    # Subject 070814-1 has an unexpected task: Video2
    # Subject 070814-1 has an unexpected task: Closed
    # Subject 070814-1 has 1 files for task Closed1, expected 2.
    # Subject 070814-1 has issues with tasks.

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=True)

    print_jordan_notes(this_subject)
    # id: 070814-1
    # pla: ('20140821', "# on disk. 'Closed1' was named 'Closed'")
    # lsd: ('20140807', "# on disk. There is a second Video named 'Video2'.")

    index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['session'] == '20140821') & (df_megs['task'] == 'Closed')].index
    df_megs_corrected.loc[index, 'task'] = 'Closed1'


    # Actually, according to Venkatesh, use 'Video2' instead of Video (Video gives hpi info missing error)

    index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['session'] == '20140807') & (df_megs['task'] == 'Video')].index

    df_megs_corrected = df_megs_corrected.drop(index)

    index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['session'] == '20140807') & (df_megs['task'] == 'Video2')].index

    # rename Video2 to Video
    df_megs_corrected.loc[index, 'task'] = 'Video'

    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

    ##################################################
    # Subject 050913-1 has an unexpected task: Open
    # Subject 050913-1 has 1 files for task Open1, expected 2.
    # Subject 050913-1 has issues with tasks.

    this_subject = '050913-1'
    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

    df_megs[df_megs['subjectNumber'] == this_subject]
    # id: 050913-1
    # pla: ('20140612', '# on disk')
    # lsd: ('20140529', "# on disk. 'Open1' was saved as 'Open'")

    index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['session'] == '20140529') & (df_megs['task'] == 'Open')].index
    df_megs_corrected.loc[index, 'task'] = 'Open1'


    # Now apply session placebo/lsd logic
    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

    #######################################################################
    # Checking subject 240107-6
    # Subject 240107-6 has an unexpected task: Closed12
    # Subject 240107-6 has issues with tasks.

    this_subject = '240107-6'

    print_jordan_notes(this_subject)

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
    # drop the task Closed12
    index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['task'] == 'Closed12')].index

    df_megs_corrected = df_megs_corrected.drop(index)

    #now apply session placebo/lsd logic

    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)


    ########################################################################
    # Checking subject 041213-1
    # Subject 041213-1 has an unexpected task: Audio
    # Subject 041213-1 has 1 files for task Music, expected 2.
    # Subject 041213-1 has issues with tasks.

    this_subject = '041213-1'
    print_jordan_notes(this_subject)

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

    index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['task'] == 'Audio')].index

    # replace Audio with Music
    df_megs_corrected.loc[index, 'task'] = 'Music'

    # Now apply session placebo/lsd logic

    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

    ########################################################################

    # Checking subject 230911-1
    this_subject = '230911-1'
    print_jordan_notes(this_subject)
    # id: 230911-1
    # pla: ('20140515', "# on disk. 'Closed2' was named simply 'Closed'.")
    # lsd: ('20140501', "# id was misnamed 231109-1. There is a second music rec named 'Music2'.")

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

    other_alias = '231109-1'

    check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)

    # rename the subjectNumber to the correct id

    indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
    df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject

    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

    index = df_megs_corrected[(df_megs_corrected['subjectNumber'] == this_subject) & (df_megs_corrected['task'] == 'Music2')].index
    df_megs_corrected = df_megs_corrected.drop(index)

    index = df_megs_corrected[(df_megs_corrected['subjectNumber'] == this_subject) & (df_megs_corrected['session'] == '20140515') & (df_megs_corrected['task'] == 'Closed')].index
    df_megs_corrected.loc[index, 'task'] = 'Closed2'

    # Now apply session placebo/lsd logic
    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)
    #############################################

    # Checking subject 010514-1
    # Subject 010514-1 has 1 files for task Video, expected 2.
    # Subject 010514-1 has 1 files for task Open1, expected 2.
    # Subject 010514-1 has 1 files for task Open2, expected 2.
    # Subject 010514-1 has 1 files for task Closed1, expected 2.
    # Subject 010514-1 has 1 files for task Closed2, expected 2.
    # Subject 010514-1 has 1 files for task Music, expected 2.
    # Subject 010514-1 has issues with tasks.

    this_subject = '010514-1'
    print_jordan_notes(this_subject)
    # id: 010514-1
    # pla: ('20140501', '# on disk')
    # lsd: ('20140515', "# id was misnamed 300414-1 in ds ('LSD Analysis.xlsx' misreports this mistake)")

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

    other_alias = '300414-1'
    check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)

    # rename the subjectNumber to the correct id
    indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
    df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject

    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

    # now apply session placebo/lsd logic

    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)


    ##################################################################################
    # Checking subject 140514-1
    # Subject 140514-1 has an unexpected task: Music2
    # Subject 140514-1 has issues with tasks.

    this_subject = '140514-1'
    print_jordan_notes(this_subject)

    # id: 140514-1
    # pla: ('20140528', '# on disk')
    # lsd: ('20140514', "# on disk. There is a second music rec named 'Music2'.")

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
    index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['task'] == 'Music2')].index
    df_megs_corrected = df_megs_corrected.drop(index)

    # Now apply session placebo/lsd logic
    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    ##################################################################################

    # Checking subject 090714-2
    # Subject 090714-2 has 1 files for task Video, expected 2.
    # Subject 090714-2 has 1 files for task Open1, expected 2.
    # Subject 090714-2 has 1 files for task Open2, expected 2.
    # Subject 090714-2 has 1 files for task Closed1, expected 2.
    # Subject 090714-2 has 1 files for task Closed2, expected 2.
    # Subject 090714-2 has 1 files for task Music, expected 2.
    # Subject 090714-2 has issues with tasks.

    this_subject = '090714-2'
    print_jordan_notes(this_subject)

    # id: 090714-2
    # pla: ('20140709', '# id was misnamed 070914-2 in ds')
    # lsd: ('20140730', '# on disk')

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
    other_alias = '070914-2'
    check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)
    # rename the subjectNumber to the correct id
    indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
    df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject
    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

    # now apply session placebo/lsd logic
    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    ##################################################################################

    # Checking subject 010813-4
    # Subject 010813-4 has 1 files for task Video, expected 2.
    # Subject 010813-4 has 1 files for task Open1, expected 2.
    # Subject 010813-4 has 1 files for task Open2, expected 2.
    # Subject 010813-4 has 1 files for task Closed1, expected 2.
    # Subject 010813-4 has 1 files for task Closed2, expected 2.
    # Subject 010813-4 has 1 files for task Music, expected 2.
    # Subject 010813-4 has issues with tasks.

    this_subject = '010813-4'
    print_jordan_notes(this_subject)

    # id: 010813-4
    # pla: ('20140820', '# on disk')
    # lsd: ('20140903', '# id was misnamed 010814-4 in ds')

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)

    other_alias = '010814-4'

    check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)

    # rename the subjectNumber to the correct id
    indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
    df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject
    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

    # now apply session placebo/lsd logic
    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    ##################################################################################
    # Checking subject 140514-2
    # Subject 140514-2 has 1 files for task Video, expected 2.
    # Subject 140514-2 has 1 files for task Open1, expected 2.
    # Subject 140514-2 has 1 files for task Open2, expected 2.
    # Subject 140514-2 has 1 files for task Closed1, expected 2.
    # Subject 140514-2 has 1 files for task Closed2, expected 2.
    # Subject 140514-2 has 1 files for task Music, expected 2.
    # Subject 140514-2 has issues with tasks.

    this_subject = '140514-2'
    print_jordan_notes(this_subject)

    # id: 140514-2
    # pla: ('20140514', '# id was misnamed 280212-2 in ds')
    # lsd: ('20140528', '# on disk')

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
    other_alias = '280212-2'
    check_subject_tasks(other_alias, df_megs, correct_tasks=correct_tasks, print_df=False)

    # rename the subjectNumber to the correct id
    indexes = df_megs_corrected[df_megs_corrected['subjectNumber'] == other_alias].index
    df_megs_corrected.loc[indexes, 'subjectNumber'] = this_subject
    check_subject_tasks(this_subject, df_megs_corrected, correct_tasks=correct_tasks, print_df=False)

    # now apply session placebo/lsd logic
    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)
    ##################################################################################

    # Checking subject 270613-1
    # Subject 270613-1 has an unexpected task: Opoen1
    # Subject 270613-1 is missing task: Music
    # Subject 270613-1 has 1 files for task Open1, expected 2.
    # Subject 270613-1 has 0 files for task Music, expected 2.
    # Subject 270613-1 has issues with tasks.

    this_subject = '270613-1'
    print_jordan_notes(this_subject)

    # id: 270613-1
    # pla: ('20140625', "# on disk. 'Opoen1' instead of 'Open1'. Music missing.")
    # lsd: ('20140611', '# on disk. Music missing.')

    check_subject_tasks(this_subject, df_megs, correct_tasks=correct_tasks, print_df=False)
    index = df_megs[(df_megs['subjectNumber'] == this_subject) & (df_megs['task'] == 'Opoen1')].index
    df_megs_corrected.loc[index, 'task'] = 'Open1'
    # Now apply session placebo/lsd logic
    df_megs_corrected = placebo_lsd_logic(df_megs_corrected, jordan_df, this_subject)

    # Missing music task expected...


    bads,_=check_all_subjects(df_megs_corrected, correct_tasks=correct_tasks)

    # set(bads).intersection(set(list(jordan_df['id'].values)))
    # that was used to identify unexpected subjects (outside of jordan notes)

    print(f"Bad subjects: {bads}")

    # restore fullpaths
    df_megs_corrected['file'] = df_megs_corrected['file'].apply(lambda x: os.path.join(SOURCE_PATH, x))

    df_megs = df_megs_corrected.copy()


    def valid_subject(subject_number, df_ids=df_ids):
        return subject_number in df_ids['ID'].values

    df_megs['valid_subject'] = df_megs['subjectNumber'].apply(valid_subject)

    assert df_megs['valid_subject'].all(), "There are invalid subjects in the DataFrame."

    def get_subject_label(subject_number, df_ids=df_ids):
        if subject_number in df_ids['ID'].values:
            return df_ids[df_ids['ID'] == subject_number]['label'].values[0]
        else:
            return False

    df_megs['subject_label'] = df_megs['subjectNumber'].apply(get_subject_label)
    os.makedirs(bids_path, exist_ok=True)
    df_megs.to_csv(os.path.join(bids_path,'bids_conversion.csv'), index=False)
    return df_megs

def lsd_bids_conversion(source_path, bids_path, DATASET_CFG, pipeline_cfg, df_megs):


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  # Adjust width as needed

    df = df_megs.copy()
    BIDS_ROOT = bids_path
    os.makedirs(BIDS_ROOT, exist_ok=True)

    errors = []

    for i, row in df.iterrows():
        subject = row['subject_label']
        session = row['session']
        task = row['task']
        filepath = row['file']

        try:
            #breakpoint()
            bidsTree = BIDSPath(subject=subject, session=session, task=task, root=BIDS_ROOT, datatype='meg', suffix='meg', extension='.fif')
            # i dont know why this level of specificity in bids arguments sometimmes is not needed
            if not os.path.isfile(bidsTree.fpath):
                mne_data = mne.io.read_raw(filepath, preload=True)

                write_raw_bids(mne_data, bids_path=bidsTree, overwrite=True, format="FIF", allow_preload=True)
            else:
                print(f"File {bidsTree.fpath} already exists, skipping.")

            print(f"Processed {i+1}/{len(df)}: {subject}, {session}, {task}, {filepath}")
        except Exception as e:
            print(f"Error processing {i+1}/{len(df)}: {subject}, {session}, {task}, {filepath}")
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            errors.append({
                'index': i,
                'subject': subject,
                'session': session,
                'task': task,
                'filepath': filepath,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            continue

    # Save errors to a CSV file for later inspection
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_csv = os.path.join(BIDS_ROOT, 'bids_errors.csv')
        errors_df.to_csv(errors_csv, index=False)
        print(f"Errors saved to {errors_csv}")


def loadmat(x,kwargs={}):
    print(f"Loading {x} with scipy.io.loadmat")
    return sio.loadmat(x,**kwargs)


def get_metadata(meg, filepath=None):
    try:
        #kwargs={'squeeze_me': True, 'struct_as_record': False})
        #squeeze_me=True, struct_as_record=False)
        data_dict = meg['data']  # Assuming 'data' is the key for the main data structure
        # 1. Extract data
        fsample = data_dict['fsample']  # e.g., 1200
        labels = data_dict['label']     # list of channel names
        data_shape = data_dict['trial'].shape    # assuming continuous → one array of shape (n_channels, n_samples)
        times = data_dict['time']    # should be 1D array of length n_samples
        try:
            event = data_dict['cfg']['event']  # Dict with keys: 'duration', 'offset', 'sample', 'type', 'value'
            trl = data_dict['cfg']['trl']      # Usually a numpy array with shape (n_trials, 3)
        except:
            event = None
            trl = None
        try:
            cfg_keys = list(data_dict['cfg'].keys())  # Get all keys in the cfg dictionary
        except:
            cfg_keys = None
        # make dictionary with this information

        metadata = {
            'filepath': filepath if filepath else 'not provided',
            'fsample': fsample,
            'labels': labels,
            'data_shape': data_shape,
            'times': times,
            'event': event,
            'trl': trl,
            'cfg_keys': cfg_keys
        }
        return metadata
    except Exception as e:
        metadata = {
            'filepath': filepath if filepath else 'not provided',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return metadata


def inspect_meg_data(mats, file='meg_data.txt'):
    amount = len(mats)
    metadatas = []
    for i, meg_path in enumerate(mats):
        print(f"Processing {i+1}/{amount}: {meg_path}")
        meg_data = loadmat(meg_path, kwargs=dict(simplify_cells=True))
        metadata = get_metadata(meg_data, meg_path)
        with open(file, 'a') as f:
            pprint(metadata, stream=f)
        metadatas.append(metadata)
    return metadatas


def fieldtrip_to_bids(source_path, bids_path, DATASET_CFG, pipeline_cfg):
    MOUNT = pipeline_cfg.get('mount', None)
    this_dataset = DATASET_CFG.get('dataset_label','Nolabel')
    filepath = os.path.join(bids_path, f'meg_{this_dataset}_metadata.csv')
    filepath_pkl = os.path.join(bids_path, f'meg_{this_dataset}_metadata.pkl')
    #breakpoint()
    if not os.path.exists(filepath_pkl):
        print(f"File {filepath} does not exist, inspecting datasets...")
        FILES_PER_DATASET = None  # Number of files to inspect per dataset
        metadatas_dict = {}
        dataset_name = this_dataset
        dataset_path = source_path
        print(f"Inspecting dataset: {dataset_name} at {dataset_path}")
        mats = glob.glob(os.path.join(dataset_path, '**', '*.mat'), recursive=True)
        root = bids_path
        os.makedirs(root, exist_ok=True)

        print(f"Found {len(mats)} .mat files in {dataset_path}, limiting to {FILES_PER_DATASET} files for inspection.")

        if FILES_PER_DATASET is not None:
            mats = mats[:FILES_PER_DATASET]  if len(mats) > FILES_PER_DATASET else mats

        metadatas_dict[dataset_name] = []
        #filepath = os.path.join(root, f'{dataset_name}_meg_data.txt') # defined already outside the loop
        metadatas = inspect_meg_data(mats, file=filepath)
        metadatas_dict[dataset_name] = metadatas
        print(f"Inspection results saved to {filepath}")

        df = pd.DataFrame.from_records(metadatas_dict[dataset_name])

        df.to_csv(filepath, index=False, sep=';', encoding='utf-8')
        df.to_pickle(filepath_pkl)
    else:
        print(f"File {filepath} already exists, loading metadata from CSV...")
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    #breakpoint()
    df['filepath'].iloc[0]
    df['data_shape']
    df['fsample'].value_counts()
    df.columns
    #df['event'].apply(lambda x: len(x) if x is not None else 0).value_counts()
    df['event'].iloc[0]

    for i, row in df.iterrows():
        print(i,row['event'])


    # Inspect errors
    errors  = df[pd.isna(df['fsample'])]

    df['cfg_keys'].iloc[0]

    for i, row in errors.iterrows():
        print(row['filepath'])
        print(row['error'])
        print(row['traceback'])

    # '/home/yorguin/scratch/data/MEG_perampanel/meg_data/PMP_PMP_020414_50.mat'
    if this_dataset not in ['psilocybin']:
        pattern = os.path.join(source_path, r'%ignore%_%session%_%subject%_%number%.mat')
    else:
        pattern = os.path.join(source_path,r"%ignore%" ,r'%session%_%subject%_%number%.mat')

    bids_items = []
    #breakpoint()
    for i, row in df.iterrows():
        bids_dict = parse_from_placeholder(row['filepath'],pattern)
        filepath = row['filepath']
        bids_dict['filepath'] = filepath
        bids_items.append(bids_dict)


    df_bids = pd.DataFrame(bids_items)

    df_bids['subject'].unique().shape

    df_bids = df_bids.sort_values(by=['subject', 'session'])

    df_bids['filename'] = df_bids['filepath'].apply(lambda x: os.path.basename(x))

    df_bids['label'] = 'S' + df_bids['subject'] + 'N' + df_bids['number']

    df_bids['session_bids'] = df_bids['session'].apply(lambda x: 'placebo' if x == 'PLA' else this_dataset) # or maybe just put drug here?, but then we need to change this also on the conversion of lsd



    # Most of them have 2 events, or 0.
    # Will convert as is, to a raw file

    BIDS_ROOT = bids_path
    os.makedirs(BIDS_ROOT, exist_ok=True)
    errors = []
    for i, row in df_bids.iterrows():
        try:
            print(f"Processing {i+1}/{len(df)}: {row['filepath']}")
            print(row)


            subject = row['label']
            session = row['session_bids']
            task = 'resting'
            filepath = row['filepath']
            os.makedirs(BIDS_ROOT, exist_ok=True)
            #breakpoint()

            bidsTree = BIDSPath(subject=subject, session=session, task=task, root=BIDS_ROOT, datatype='meg', suffix='meg', extension='.fif')

            if not os.path.isfile(bidsTree.fpath):


                meg_path = row['filepath']
                meg_data = loadmat(meg_path, kwargs=dict(simplify_cells=True))
                metadata = get_metadata(meg_data, meg_path)

                chantype_map = {
                'meggrad': 'grad',     # MNE type for gradiometers
                'refgrad': 'grad',     # if you want to treat refgrad as grad
                'refmag': 'mag'        # or maybe 'misc' if they are not standard MEG
                }


                ch_names = meg_data['data']['label'].tolist()
                types = [chantype_map.get(ch, 'misc') for ch in ch_names]  # Default to 'misc' if not found
                chantypes = ['meggrad', 'meggrad', 'refmag', 'refgrad']
                mne_types = [chantype_map.get(t, 'misc') for t in chantypes]

                # Create a dictionary to pass to set_channel_types
                ch_type_dict = {name: typ for name, typ in zip(ch_names, mne_types)}

                # Set types
                if this_dataset == 'psilocybin':
                    #breakpoint()
                    #pass
                    # Find 'InfusionStop'
                    # Find 'RestStop'
                    infusion_stop = None
                    rest_stop = None
                    infusion_start = None
                    rest_start = None
                    for e in metadata['event']:
                        if e['type'] == 'InfusionStart':
                            infusion_start = e['sample']
                        if e['type'] == 'RestStart':
                            rest_start = e['sample']
                        if e['type'] == 'InfusionStop':
                            infusion_stop = e['sample']
                        elif e['type'] == 'RestStop':
                            rest_stop = e['sample']
                        if infusion_stop is not None and rest_stop is not None:
                            break
                    if infusion_stop is None or rest_stop is None:
                        raise ValueError("InfusionStop or RestStop event not found in metadata.")
                    # Create a RawArray with the data and info
                    raw = mne.io.RawArray(
                        data=meg_data['data']['trial'],  # Assuming 'trial' is a 2D array with shape (n_channels, n_samples)
                        info=mne.create_info(
                            ch_names=meg_data['data']['label'].tolist(),  # Assuming 'label' is a list of channel names
                            sfreq=meg_data['data']['fsample'],  # Assuming 'fsample' is a scalar
                            ch_types=types  # Assuming all channels are meg
                        ))

                    # Add the events to the info
                    #events = np.array([[infusion_stop, 0, 1], [rest_stop, 0, 2]])?

                    # or just crop between these samples?
                    #breakpoint()
                    # actually they cutted the data from RestStart to RestStop
                    # you can check that
                    # num_samples = meg_data['data']['trial'].shape[1]
                    # rest_stop - rest_start == num_samples - 1
                    # so we will substract rest_start from both infusion_stop and rest_stop
                    infusion_sec = raw.times[infusion_stop-rest_start]
                    rest_sec = raw.times[rest_stop-rest_start]  # Convert sample index to seconds
                    print(f"Cropping between: InfusionStop at {infusion_sec}, RestStop at {rest_sec}, in seconds.")
                    print(f"Delta time: {(rest_sec - infusion_sec)/60} min = {rest_sec - infusion_sec} seconds = {rest_stop - infusion_stop} samples.")
                    raw = raw.crop(tmin=infusion_sec, tmax=rest_sec, include_tmax=True)  # Crop the raw data between the two events
                else:
                    raw = mne.io.RawArray(
                        data=meg_data['data']['trial'],  # Assuming 'trial' is a 2D array with shape (n_channels, n_samples)
                        info=mne.create_info(
                            ch_names=meg_data['data']['label'].tolist(),  # Assuming 'label' is a list of channel names
                            sfreq=meg_data['data']['fsample'],  # Assuming 'fsample' is a scalar
                            ch_types=types  # Assuming all channels are meg
                        )
                    )

                raw.set_channel_types(ch_type_dict)


                write_raw_bids(raw, bids_path=bidsTree, overwrite=True, format="FIF", allow_preload=True)
            else:
                print(f"File {bidsTree.fpath} already exists, skipping.")

        except Exception as e:
            print(f"Error processing {i+1}/{len(df)}: {row['filepath']}")
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            errors.append({
                'index': i,
                'subject': row['subject'],
                'session': row['session'],
                'filepath': row['filepath'],
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    # Save errors to a CSV file for later inspection
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_csv = os.path.join(BIDS_ROOT, 'bids_errors.csv')
        errors_df.to_csv(errors_csv, index=False)



def bidsify(source_path, bids_path, DATASET_CFG,pipeline_cfg):
    """
    Convert source_path to BIDS format and save to bids_path.
    
    Parameters
    ----------
    source_path : str
        Path to the source data.
    bids_path : str
        Path where the BIDS dataset will be saved.
    DATASET_CFG : dict
        Configuration dictionary for the dataset.
    """
    print(f"Converting {source_path} to BIDS format at {bids_path}")
    
    # Example of how you might use DATASET_CFG
    rule = DATASET_CFG.get('bidsify', {}).get('pattern', None)
    
    import glob, os, pathlib
    #breakpoint()
    print(DATASET_CFG.get('dataset_label','Nolabel'))
    if DATASET_CFG.get('dataset_label','') == 'lsd': # You could add per dataset handling here
        df_megs = thanks_jordan_venkatesh(source_path, bids_path, DATASET_CFG, pipeline_cfg)
        lsd_bids_conversion(source_path, bids_path, DATASET_CFG, pipeline_cfg, df_megs)


    if DATASET_CFG.get('dataset_label','') in ['perampanel','psilocybin']:
        fieldtrip_to_bids(source_path, bids_path, DATASET_CFG, pipeline_cfg)

def parse_bids(bidsname):
    name = os.path.basename(bidsname)
    entities=name.split('_')
    suffix = entities[-1]
    ext = suffix.split('.')[-1]
    suffix = suffix.split('.')[0]
    entities = entities[:-1]
    d={}
    for item in entities:
        l=item.split('-')
        key=l[0]
        val=l[1]
        d[key]=val
    if not '-' in suffix:
        d['suffix']=suffix
    else:
        a,b=suffix.split('-')
        d[a]=b
    return d

# signature prepare(filename=raw_file, dataset_cfg, njobs=njobs, **this_prep['prepare'])
def prepare(filename, dataset_cfg=None, njobs=1, downsample = 500, normalization = False, filter_args=None, epoch_config={}):
    """
    keep_chans: is ignored, only used to keep the same signature as the original function
    line_noise: is ignored, only used to keep the same signature as the original function
    njobs: is ignored, only used to keep the same signature as the original function
    """
    info = {}
    figures = []

    info['filename'] = filename
    info['dataset_cfg'] = dataset_cfg
    info['downsample'] = downsample
    info['normalization'] = normalization
    info['filter_args'] = filter_args
    info['njobs'] = njobs
    info['epoch_config'] = epoch_config

    eegpath = filename
    print('PREPARE FUNCTION OVERRIDENED')


    raw = mne.io.read_raw(eegpath,verbose=False,preload=True)

    if normalization:
        # It is debatable where to normalize the data. Here we do it after PyPREP.
        # Sanity check, the argmin of the zscored data should be the same as the argmin of the raw data
        assert np.argmin(raw.get_data()[0,:])==np.argmin(scipy.stats.zscore(raw.get_data(),axis=1)[0,:])
        raw._data = scipy.stats.zscore(raw.get_data(),axis=1)
        print('AMPLITUDE NORMALIZATION DONE')

    # Filter the data
    if filter_args is not None:
        raw = raw.filter(**filter_args,verbose=False)
        print('FILTERED',end=' ')

    # Extract epochs
    print('EPOCH SEGMENTATION')
    epochs = mne.make_fixed_length_epochs(raw,preload=True,**epoch_config)

    if downsample is not None:
        epochs = epochs.resample(downsample)


    return epochs,info,figures, None
