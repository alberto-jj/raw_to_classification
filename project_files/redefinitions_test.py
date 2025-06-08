from mne.io import read_raw
from mne import read_epochs
def hello(name='World'):
    return f"Hello, {name}!"

def load_meeg(meeg_file, kwargs={}):
    """Load a MEEG file using MNE-Python."""
    print('Example redefinition of load_meeg function')
    hello('MEEG World')
    try:
        meeg = read_raw(meeg_file, **kwargs)
    except Exception as e:
        # try to load as epochs
        try:
            meeg = read_epochs(meeg_file, **kwargs)
        except Exception as e:
            raise ValueError(f"Could not load MEEG file {meeg_file}. Error: {e}")
    return meeg
