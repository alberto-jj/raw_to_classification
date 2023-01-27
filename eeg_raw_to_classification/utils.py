import base64
from io import BytesIO
import json
import yaml
import numpy as np
from copy import deepcopy
import os
import itertools

# Get the derivatives path in BIDS format
def get_derivative_path(layout,eeg_file,output_entity,suffix,output_extension,bids_root,derivatives_root):
    entities = layout.parse_file_entities(eeg_file)
    derivative_path = eeg_file.replace(bids_root,derivatives_root)
    derivative_path = derivative_path.replace(entities['extension'],'')
    derivative_path = derivative_path.split('_')
    desc = 'desc-' + output_entity
    derivative_path = derivative_path[:-1] + [desc] + [suffix]
    derivative_path = '_'.join(derivative_path) + output_extension 
    return derivative_path

# Save FIGS
def save_figs_in_html(htmlfile,figures):
    htmls = ['<img src=\'data:image/png;base64,{}\'>'.format(imgfoo(fig)) for fig in figures]
    html = "\n".join(htmls)
    with open(htmlfile,'w') as f:
        f.write(html)

def imgfoo(fig):
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return encoded

def save_dict_to_json(jsonfile,data):
    with open(jsonfile, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def parse_bids(bidsname):
    entities=bidsname.split('_')
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
    d['suffix']=suffix
    return d

def load_yaml(rules):
    """Load rules if given a path, bypass if given a dict.
    Parameters
    ----------
    
    rules : str|dict
        The path to the rules file, or the rules dictionary.
    Returns
    -------
    dict
        The rules dictionary.
    """
    if isinstance(rules,str):
        try:
            with open(rules,encoding="utf-8") as f:
                return yaml.load(f,yaml.FullLoader)
        except:
            raise IOError(f"Couldnt read {rules} file as a rule file.")
    elif isinstance(rules,dict):
        return deepcopy(rules)
    else:
        raise ValueError(f'Expected str or dict as rules, got {type(rules)} instead.')

def get_output_dict(eeg_file,FORMAT='WIDE',dataset_label='',feature_suffix=''):
    output = np.load(eeg_file,allow_pickle=True).item()
    filename = os.path.basename(eeg_file)
    subject = parse_bids(filename)['sub']
    task = parse_bids(filename)['task']
    dataset = dataset_label
    # Assume python > 3.7, dictionaries retain order
    axes = list(output['metadata']['axes'].values())
    keys =list(output['metadata']['axes'].keys())
    
    dict_list = []
    d = {'dataset':dataset,'subject':subject,'task':task}

    for combination in itertools.product(*axes):
        indexes = []
        for i,j in enumerate(combination):
            indexes.append(axes[i].index(j))

        value = eval(f'output["values"]{indexes}')
        if FORMAT == 'LONG':
            d = {'subject':subject,'dataset':dataset}
            for key,val in zip(keys,combination):
                d[feature_suffix+key]=val
            d['value']=value
            dict_list.append(d)
        elif FORMAT == 'WIDE':
            final_key = ''
            first = True
            for key,val in zip(keys,combination):
                if first:
                    final_key += val
                    first=False
                else:
                    final_key += '.'+val
            #final_key = '_'.join(final_key)
            d[feature_suffix+final_key]=value
    if FORMAT=='WIDE':
        dict_list.append(d)
    return dict_list
