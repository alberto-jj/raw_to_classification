from copy import deepcopy

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
        extra_metadata['provenance'] = [deepcopy(input.filenames)] # for Raw is a list of strings, but we will keep them as a unit
        extra_metadata['input_type'] = 'Raw'
    if len(input.get_data().shape) == 3:
        input_order =  ('epochs', 'spaces', 'times')
        input_axes['epochs'] = list(range(input.get_data().shape[0]))
        extra_metadata['provenance'] = [deepcopy(input.filename)] # for Epochs is singular, a string
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

