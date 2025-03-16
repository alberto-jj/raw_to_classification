import mne
import os
import glob
import argparse
from eeg_raw_to_classification.utils import load_yaml, save_dict_to_json, save_figs_in_html
import numpy as np
import pandas as pd
import traceback

def main(pipeline_file):
    cfg = load_yaml(pipeline_file)
    PROJECT = cfg['project']
    datasets = load_yaml(cfg['datasets_file'])

    inspect_path = cfg['inspect']['path'].replace('%PROJECT%', PROJECT)
    os.makedirs(inspect_path, exist_ok=True)

    max_files = None  # 30
    for dslabel, DATASET in datasets.items():
        if DATASET.get('skip', False):
            continue
        exemplar_file = DATASET['example_file']
        eegs = glob.glob(exemplar_file, recursive=True)
        MONTAGES = []
        TIMES = []
        EEGs = []
        for i, eeg in enumerate(eegs):
            filename = os.path.basename(eeg)
            output_base = os.path.join(inspect_path, dslabel, dslabel + '-' + filename)
            os.makedirs(os.path.dirname(output_base), exist_ok=True)
            try:
                raw = mne.io.read_raw(eeg)  # inspect line noise freq and channel names
                TIMES.append(raw.times[-1])
                ch_names = raw.ch_names
                MONTAGES.append(ch_names)
                EEGs.append(eeg)
                if i == 0:
                    fig = raw.plot_psd(show=False)
                    save_figs_in_html(output_base + '_spectrum.html', [fig])
            except:
                save_dict_to_json(output_base + '_problem.txt', {'problem': traceback.format_exc()})
                print(traceback.format_exc())
            if max_files and i > max_files:
                break
        common = set(MONTAGES[0])
        union_montage = set(MONTAGES[0])
        for montage in MONTAGES[1:]:
            common = common.intersection(set(montage))
            union_montage = union_montage.union(set(montage))

        save_dict_to_json(os.path.join(inspect_path, dslabel, 'common_montage.txt'), {'common_montage': list(common)})
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'union_montage.txt'), {'union_montage': list(union_montage)})

        save_dict_to_json(os.path.join(inspect_path, dslabel, 'times.txt'), {'times': TIMES})
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'times_stats.txt'), {'mean': np.mean(TIMES), 'max': np.max(TIMES), 'min': np.min(TIMES), 'median': np.median(TIMES), 'std': np.std(TIMES)})
        
        counts_dict = dict(zip(*np.unique(TIMES, return_counts=True)))
        counts_dict = {int(k): int(v) for k, v in counts_dict.items()}  # Convert keys and values to Python int
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'times_counts.txt'), {'counts': counts_dict})

        df = pd.DataFrame({'EEG': EEGs, 'montage': MONTAGES, 'times': TIMES})
        df.to_csv(os.path.join(inspect_path, dslabel, f'{dslabel}_inspect.csv'))
        print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect EEG datasets.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline.yml file')
    args = parser.parse_args()
    main(args.pipeline_file)