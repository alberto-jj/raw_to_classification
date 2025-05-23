
from copy import deepcopy
from mne.io import Raw
from mne import Epochs
import numpy as np
from copy import deepcopy
import json
import pickle
from mne.io import read_raw
from mne import read_epochs
def primitive_feature_to_format(this_type:str):
    """Convert a feature type to a standardized format.

    Parameters
    ----------
    this_type : str
        The input feature type.

    Returns
    -------
    str
        The converted feature type in a standardized format.
    """
    if this_type == 'array':
        return 'npy'
    elif this_type == 'html':
        return 'html'
    elif this_type == 'dict':
        return 'json'
    elif this_type == 'pickle':
        return 'pickle'
    else:
        raise ValueError(f"Unknown feature type: {this_type}")

def primitive_save(output, outputfile, output_format):
    """Save the output in the specified format.

    Parameters
    ----------
    output : any
        The output to save.
    outputfile : str
        The filename to save the output to.
    output_format : str
        The format to save the output in.
    """
    if output_format == 'fif':
        if isinstance(output, Raw) or isinstance(output, Epochs):
            output.save(outputfile, overwrite=True)
        else:
            raise ValueError(f"Unknown MNE object type: {type(output)}")
    if output_format == 'npy':
        np.save(outputfile,output)
    elif output_format == 'json':
        with open(outputfile, 'w') as f:
            json.dump(output, f, indent=4)
    elif output_format == 'html':
        with open(outputfile, 'w') as f:
            f.write(output)
    elif output_format == 'pickle':
        with open(outputfile, 'wb') as f:
            pickle.dump(output, f)
    else:
        raise ValueError(f"Unknown format: {output_format}")

def primitive_load(outputfile, output_format):
    """Load the output from the specified format.

    Parameters
    ----------
    outputfile : str
        The filename to load the output from.
    output_format : str
        The format to load the output in.

    Returns
    -------
    any
        The loaded output.
    """
    if output_format == 'fif':
        try:
            return read_raw(outputfile, preload=True)
        except:
            return read_epochs(outputfile, preload=True)

    if output_format == 'npy':
        return np.load(outputfile,allow_pickle=True).item()
    elif output_format == 'json':
        with open(outputfile, 'r') as f:
            return json.load(f)
    elif output_format == 'html':
        with open(outputfile, 'r') as f:
            return f.read()
    elif output_format == 'pickle':
        with open(outputfile, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {output_format}")

def snake_to_camel(snake_str):
    """Convert a snake_case string to CamelCase.

    Parameters
    ----------
    snake_str : str
        The input string in snake_case format.

    Returns
    -------
    str
        The converted string in CamelCase format.
    """
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def get_mne_metadata(input, save_input_no_data=False):
    """Get the metadata from an MNE object (Raw or Epochs) and return it in a standardized format.

    Parameters
    ----------
    input : Raw or Epochs
        The MNE object to extract metadata from.
    save_input_no_data : bool
        If True, save a copy of the input object with no data to save space.
        If False, do not save the copy.
        Default is False.
        Note that even if save_input_no_data is False, the input object can still take around 3 MB of memory for example.
        In any case, the filename of input is saved so that you can load those metadata later if needed.
    Returns
    -------
    input_order : tuple
        The order of the axes in the input object.
        For example, ('epochs', 'spaces', 'times') for Epochs.
    """
    input_axes = {}
    input_order = ()
    extra_metadata = {}

    if len(input.get_data().shape) == 2:
        input_order = ('spaces', 'times')
        extra_metadata['filename'] = deepcopy(input.filenames) # for Raw is a list of strings
        extra_metadata['input_type'] = 'Raw'
    if len(input.get_data().shape) == 3:
        input_order =  ('epochs', 'spaces', 'times')
        input_axes['epochs'] = list(range(input.get_data().shape[0]))
        extra_metadata['filename'] = deepcopy(input.filename) # for Epochs is singular, a string
        extra_metadata['input_type'] = 'Epochs'

    if save_input_no_data:
        input_no_data = input.copy()
        input_no_data._data = None
        input_no_data.preload = False
        extra_metadata['input_no_data'] = input_no_data

        # input_no_data.info has the 
        # montage : info.get_montage()
        # sfreq
        # ch_types : info.get_channel_types()
        # and other metadata
        # you can use data from input_no_data to get the metadata
        # only the data is not there to save space
        # obviusly, some methods will not work...

    else:
        extra_metadata['input_no_data'] = None


    input_axes['spaces'] = deepcopy(input.ch_names)
    input_axes['times'] = deepcopy(input.times)

    return input_order, input_axes, extra_metadata

