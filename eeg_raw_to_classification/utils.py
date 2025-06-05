import base64
from io import BytesIO
import json
import yaml
import numpy as np
from copy import deepcopy
import os
import itertools
import pathlib
import copy

def extract_item(data,fun,newtype):
    if isinstance(fun,str):
        fun=eval(fun.replace('eval%',''))
    data = copy.deepcopy(data)
    newfoo=np.vectorize(fun)#, otypes=[object])
    data['values'] = newfoo(data['values'])
    if newtype:
        data['metadata']['type'] = newtype
    return data
def agg_numpy(x,numpyfun,axisname='epochs',max_numitem=None): # or give a more complex indexing for items
    if isinstance(numpyfun,str):
        numpyfun=eval(numpyfun.replace('eval%',''))
    x = copy.deepcopy(x)
    # input is the dict from np.load
    # a function like this could help for rois
    # spaces gets mapped to rois for example, and you modify the metadata appropiately
    axis= x['metadata']['order']
    axis = axis.index(axisname)

    if max_numitem is not None:
        if x['values'].shape[axis] >= max_numitem:
            # if we have more than max_numitem, we take the first max_numitem
            x['values'] = np.take(x['values'], indices=range(max_numitem), axis=axis)
        else:
            print(f"Warning: {x['values'].shape[axis]} items in axis {axisname} are less than max_numitem {max_numitem}.")

    # handle metadata appropriately
    x['values'] = numpyfun(x['values'],axis=axis)
    order = list(x['metadata']['order'])
    order.remove(axisname)
    x['metadata']['order'] = tuple(order)
    del x['metadata']['axes'][axisname]
    return x

def get_path(path, MOUNT=None):
    if MOUNT and type(path)==dict:
        output = path[MOUNT]
    else:
        output = path
    return output

# Get the derivatives path in BIDS format
def get_derivative_path(layout,eeg_file,output_entity,suffix,output_extension,bids_root,derivatives_root):
    entities = layout.parse_file_entities(eeg_file)
    eeg_file = pathlib.Path(eeg_file).as_posix()
    bids_root = pathlib.Path(bids_root).as_posix()
    derivatives_root = pathlib.Path(derivatives_root).as_posix()
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

def get_output_dict(eeg_file,FORMAT='WIDE',dataset_label='',feature_suffix='', agg_fun= None,keyvalformat=False,showinfo=False):
    output = np.load(eeg_file,allow_pickle=True).item()
    filename = os.path.basename(eeg_file)
    subject = parse_bids(filename)['sub']
    task = parse_bids(filename)['task']
    dataset = dataset_label
    # Assume python > 3.7, dictionaries retain order
    axes = list(output['metadata']['axes'].values())
    keys =list(output['metadata']['axes'].keys())
    filenosuffix = eeg_file.split('_')
    filenosuffix = '_'.join(filenosuffix[:-1])  # remove suffix
    
    ## COCOSPRINT ONLY TEMPORAL FIX
    for i,ax in enumerate(axes):
        for j,axitem in enumerate(ax):
            if '-' in axitem:
                axes[i][j] = axitem.split('-')[0]
    ##############
    dict_list = []
    d = {'dataset':dataset,'subject':subject,'task':task, 'filepath':filenosuffix}
    if showinfo:
        print(eeg_file)
        print('axes:',axes)
        print(output['values'].shape)
    
    for combination in itertools.product(*axes):
        indexes = []
        for i,j in enumerate(combination):
            indexes.append(list(axes[i]).index(j)) # list for nparrays

        value = eval(f'output["values"]{indexes}')
        if FORMAT == 'LONG':
            d = {'subject':subject,'dataset':dataset,'feature':feature_suffix}
            for key,val in zip(keys,combination):
                d[key]=val
            d['value']=agg_fun(value)
            dict_list.append(d)
        elif FORMAT == 'WIDE':
            final_key = ''
            first = True
            for key,val in zip(keys,combination):
                if first:
                    if keyvalformat:
                        final_key += str(key)+'-'+str(val)
                    else:
                        final_key += str(val)
                    first=False
                else:
                    if keyvalformat:
                        final_key += '.'+str(key)+'-'+str(val)
                    else:
                        final_key += '.'+str(val)
            #final_key = '_'.join(final_key)
            if keyvalformat:
                d['feature-'+feature_suffix+final_key]=agg_fun(value)
            else:
                d[feature_suffix+final_key]=agg_fun(value)
            if showinfo:
                try:
                    showinfo=False
                    print('feature:',feature_suffix+final_key)
                    print('value:',value)
                    print('agg value:',agg_fun(value))
                except:
                    print('Error showing info', eeg_file, 'feature:',feature_suffix+final_key)
                    pass
    if FORMAT=='WIDE':
        dict_list.append(d)
    return dict_list
