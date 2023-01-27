import mne
import os
from eeg_raw_to_classification.utils import load_yaml,save_dict_to_json,save_figs_in_html
datasets = load_yaml('datasets.yml')
cfg = load_yaml('pipeline.yml')
inspect_path = cfg['inspect']['path']
os.makedirs(inspect_path,exist_ok=True)
for dslabel,DATASET in datasets.items():
    exemplar_file = DATASET['example_file']
    filename = os.path.basename(exemplar_file)
    output_base = os.path.join(inspect_path,dslabel+'-'+filename)
    raw=mne.io.read_raw(exemplar_file) #inspect line noise freq and channel names

    ch_names=raw.ch_names
    save_dict_to_json(output_base+'_chnames.txt',dict(ch_names=ch_names))
    fig = raw.plot_psd(show=False)
    save_figs_in_html(output_base+'_spectrum.html',[fig])

