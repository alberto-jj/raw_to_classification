from copy import deepcopy
import numpy as np
from mne import read_epochs
from phyid.calculate import calc_PhiID
from mne import BaseEpochs


INFORMATION_DYNAMICS_METRICS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "DownwardCausation": ["sty", "stx", "str"],
    "UpwardCausation": ["xts", "yts", "rts"],
}

IIT_METRICS = {
    "InformationStorage": ["xtx", "yty", "rtr", "sts"],
    "TransferEntropy": ["xty", "xtr", "str", "sty"],
    "CausalDensity": ["xtr", "ytr", "sty", "str", "str", "xty", "ytx", "stx"],
    "IntegratedInformation": ["rts", "xts", "sts", "sty", "str", "yts", "ytx", "stx", "xty"],
}


{
"PhiID": {
        "tau": dict(FloatParam=(5, 1, 100, dict(doc="Time lag for the PhiID algorithm"))),
        "kind": dict(StringParam=(
            "gaussian",
            dict(options=["gaussian", "discrete"]),
            dict(doc="Kind of data (continuous Gaussian or discrete-binarized)"),
        )),
        "redudancy": dict(StringParam=("MMI", dict(options=["MMI", "CCS"]), dict(doc="Redundancy measure to use"))),
    }
}
def single_atoms(epochs, tau=5,redundancy='MMI', kind='gaussian', channel_labels=None):


    if isinstance(epochs, BaseEpochs):
        #breakpoint()
        # drop channels not starting with 'M'
        # custom code for cocosprint cocodelics project
        idx_to_keep = [i for i,ch in enumerate(epochs.ch_names) if ch.startswith('M')]
        chans_to_keep = [ch for i,ch in enumerate(epochs.ch_names) if ch.startswith('M')]

        matrix = epochs.get_data()
        matrix = matrix[:, idx_to_keep, :]  # shape (n_epochs, n_channels, n_time)
        channel_labels = chans_to_keep

    else:
        # Assume input is a 3D numpy array: epochs x channels x timepoints
        matrix = np.asarray(epochs, dtype=float)
        if matrix.ndim != 3:
            raise ValueError("Input must be a 3D numpy array (epochs x channels x timepoints).")

    n_epochs, n_channels, n_time = matrix.shape

    # If channel_labels is not provided, create default labels
    if channel_labels is None:
        channel_labels = [f"ch{i}" for i in range(n_channels)]


    # List of atom names in fixed order
    atom_names = [
        "rtr",
        "rtx",
        "rty",
        "rts",
        "xtr",
        "xtx",
        "xty",
        "xts",
        "ytr",
        "ytx",
        "yty",
        "yts",
        "str",
        "stx",
        "sty",
        "sts",
    ]
    n_atoms = len(atom_names)

    atoms_vals = np.zeros((n_epochs, n_channels, n_atoms, n_time - tau), dtype=np.float64)

    # Compute PhiID for each channel vs. the mean of all other channels
    for e in range(n_epochs):
        data = matrix[e, :, :]  # shape (n_channels, n_time)
        for i in range(n_channels):
            src = data[i]
            if n_channels > 1:
                # target is average of all other channels
                trg = np.mean(data[np.arange(n_channels) != i], axis=0)
            else:
                # only one channel: create trg as the timelagged version of src
                #trg = np.roll(src, tau)
                # 
                trg = src #Antoine
                #raise ValueError("Only one channel found, cannot compute PhiID with a single channel (TODO).")

            # Run the PhiID calculation
            try:
                atoms_res, _ = calc_PhiID(src, trg, tau, kind=kind, redundancy=redundancy)
                #assert [k for k in atoms_res.keys()] == atom_names
                atoms_res['rtr'].shape
                for key in atoms_res.keys():
                    vals = atoms_res[key]
                    atoms_vals[e, i, atom_names.index(key), :] = vals# should be size [:n_time - tau]
            except Exception as ex:
                print(f"Error processing epoch {e}, channel {i} '{channel_labels[i]}': {ex}")
                # Fill with NaNs if there's an error
                atoms_vals[e, i, :, :] = np.nan



    # Build metadata
    output = {}
    output['metadata'] = {'type': 'Atoms'}

    # Define axis labels
    epoch_labels = [e for e in range(n_epochs)]
    space_names = channel_labels
    atom_names_order = atom_names
    time_axis = np.arange(n_time - tau) / epochs.info['sfreq']  # convert to seconds (optional)

    # Axes dict
    output['metadata']['axes'] = {
        'epochs': epoch_labels,
        'spaces': space_names,
        'atoms': atom_names_order,
        'times': time_axis
    }

    # Order of axes (matches shape of atoms_vals)
    output['metadata']['order'] = ('epochs', 'spaces', 'atoms', 'times')

    # Store values
    output['values'] = atoms_vals


    return output


def atoms_results(atoms, key='InformationDynamics', aggregation_mode='mean-sum'):
    """
    aggregation_mode:
        - 'sum-mean': sum across atoms at each timepoint, then mean over time (principled)
        - 'mean-sum': mean each atom first, then sum the means (to exactly match original process())
    """
    # assume atoms is a dict with keys 'values' and 'metadata' from single_atoms

    n_epochs = len(atoms['metadata']['axes']['epochs'])
    n_channels = len(atoms['metadata']['axes']['spaces'])
    n_atoms = len(atoms['metadata']['axes']['atoms'])
    n_time = len(atoms['metadata']['axes']['times'])

    assert atoms['metadata']['order'] == ('epochs', 'spaces', 'atoms', 'times')

    # Build output dict in spectrum_multitaper style
    epochs_labels = atoms['metadata']['axes']['epochs']
    space_names = atoms['metadata']['axes']['spaces']
    atom_names = atoms['metadata']['axes']['atoms']

    output = {}

    if key == 'InformationDynamics':
        the_final_vals = np.zeros((n_epochs, n_channels, len(INFORMATION_DYNAMICS_METRICS)), dtype=np.float64)
        metric_names = list(INFORMATION_DYNAMICS_METRICS.keys())
        output['metadata'] = {'type': 'InformationDynamics'}
        output['metadata']['axes'] = {
            'epochs': epochs_labels,
            'spaces': space_names,
            'metrics': metric_names
        }
        output['metadata']['order'] = ('epochs', 'spaces', 'metrics')
    elif key == 'IntegratedInformationDecomposition':
        the_final_vals = np.zeros((n_epochs,n_channels, n_atoms), dtype=np.float64)
        metric_names = atom_names
        output['metadata'] = {'type': 'IntegratedInformationDecomposition'}
        output['metadata']['axes'] = {
            'epochs': epochs_labels,
            'spaces': space_names,
            'metrics': metric_names,
        }
        output['metadata']['order'] = ('epochs', 'spaces', 'metrics')
    elif key == 'IntegratedInformationTheory':
        the_final_vals = np.zeros((n_epochs, n_channels, len(IIT_METRICS)), dtype=np.float64)
        metric_names = list(IIT_METRICS.keys())
        output['metadata'] = {'type': 'IntegratedInformationTheory'}
        output['metadata']['axes'] = {
            'epochs': epochs_labels,
            'spaces': space_names,
            'metrics': metric_names
        }
        output['metadata']['order'] = ('epochs', 'spaces', 'metrics')
    
    for e in range(n_epochs):
        for c in range(n_channels):
            for j, name in enumerate(metric_names):
                values = atoms['values'][e, c, :, :]

                if key == 'InformationDynamics':
                    # Get the indices of the atoms in the INFORMATION_DYNAMICS_METRICS dict
                    atom_indices = [atom_names.index(atom) for atom in INFORMATION_DYNAMICS_METRICS[name]]
                    # Sum the values of the atoms and average over time

                    if aggregation_mode == 'sum-mean':
                        the_final_vals[e, c, j] = float(np.mean(np.sum(values[atom_indices,:], axis=0)))
                    elif aggregation_mode == 'mean-sum':
                        the_final_vals[e, c, j] = float(np.sum([atoms['values'][e, c, atom_idx, :].mean() for atom_idx in atom_indices]))
                elif key == 'IntegratedInformationDecomposition':
                    # For PhiID, we just take the mean of the atom values
                    atom_index = atom_names.index(name)
                    # NOTE actually this two seem to be the same ??...
                    if aggregation_mode == 'sum-mean':
                        the_final_vals[e, c, j] = float(np.mean(values[atom_index, :]))
                    elif aggregation_mode == 'mean-sum':
                        the_final_vals[e, c, j] = float(atoms['values'][e, c, atom_index, :].mean())
                elif key == 'IntegratedInformationTheory':
                    atom_indices = [atom_names.index(atom) for atom in IIT_METRICS[name]]
                    # Sum the values of the atoms and average over time
                    if aggregation_mode == 'sum-mean':
                        the_final_vals[e, c, j] = float(np.mean(np.sum(values[atom_indices,:], axis=0)))
                    elif aggregation_mode == 'mean-sum':
                        the_final_vals[e, c, j] = float(np.sum([atoms['values'][e, c, atom_idx, :].mean() for atom_idx in atom_indices]))
                    if name == "Integrated information":
                        rtr_index = atom_names.index("rtr")
                        if aggregation_mode == 'sum-mean':
                            the_final_vals[e, c, j] -= float(np.mean(values[rtr_index, :]))
                        elif aggregation_mode == 'mean-sum':
                            the_final_vals[e, c, j] -= float(atoms['values'][e, c, rtr_index, :].mean())

    # Store the computed values
    output['values'] = the_final_vals
    return output


def process(matrix, tau=5, redundancy="MMI", kind="gaussian", channel_labels=None):
        # If no input, do nothing
        # Ensure data is a 2D array: channels x timepoints
        data = np.asarray(matrix, dtype=float)

        n_channels, n_time = data.shape


        # List of atom names in fixed order
        atom_names = [
            "rtr",
            "rtx",
            "rty",
            "rts",
            "xtr",
            "xtx",
            "xty",
            "xts",
            "ytr",
            "ytx",
            "yty",
            "yts",
            "str",
            "stx",
            "sty",
            "sts",
        ]
        n_atoms = len(atom_names)

        # Prepare output array: one row per channel, one col per atom
        PhiID_vals = np.zeros((n_channels, n_atoms), dtype=np.float64)
        inf_dyn_vals = np.zeros((n_channels, len(INFORMATION_DYNAMICS_METRICS)), dtype=np.float64)
        IIT_vals = np.zeros((n_channels, len(IIT_METRICS)), dtype=np.float64)
        # Compute PhiID for each channel vs. the mean of all other channels
        for i in range(n_channels):
            src = data[i]
            if n_channels > 1:
                # target is average of all other channels
                trg = np.mean(data[np.arange(n_channels) != i], axis=0)
            else:
                # only one channel: create trg as the timelagged version of src
                trg = np.roll(src, tau)
                # TODO
                raise ValueError("Only one channel found, cannot compute PhiID with a single channel (TODO).")

            # Run the PhiID calculation
            atoms_res, _ = calc_PhiID(src, trg, tau, kind=kind, redundancy=redundancy)
            # add 'str', 'stx', 'sty', 'sts' together

            # Each atoms_res[name] is a vector length n_time - tau
            # We average over time to get a single scalar per atom
            for j, name in enumerate(atom_names):
                PhiID_vals[i, j] = float(np.mean(atoms_res[name]))
            for j, name in enumerate(INFORMATION_DYNAMICS_METRICS):
                # Get the indices of the atoms in the INFORMATION_DYNAMICS_METRICS dict
                atom_indices = [atom_names.index(atom) for atom in INFORMATION_DYNAMICS_METRICS[name]]
                # Sum the values of the atoms and average over time
                inf_dyn_vals[i, j] = float(np.mean(np.sum(PhiID_vals[i, atom_indices], axis=0)))
            for j, name in enumerate(IIT_METRICS):
                # Get the indices of the atoms in the IIT_METRICS dict
                atom_indices = [atom_names.index(atom) for atom in IIT_METRICS[name]]
                # Sum the values of the atoms and average over time
                IIT_vals[i, j] = float(np.mean(np.sum(PhiID_vals[i, atom_indices], axis=0)))
                if name == "Integrated information":
                    # Subtract rtr
                    IIT_vals[i, j] -= float(np.mean(atoms_res["rtr"]))

        # Build metadata for output
        # Copy original metadata but replace channel dims
        out_meta = {}
        # Overwrite channels info
        if channel_labels is None:
            channel_labels = [f"ch{i}" for i in range(n_channels)]
        out_meta["channels"] = {"dim0": channel_labels, "dim1": atom_names}
        out_phi = {}
        out_phi["channels"] = {"dim0": channel_labels, "dim1": list(INFORMATION_DYNAMICS_METRICS.keys())}
        out_IIT = {}
        out_IIT["channels"] = {"dim0": channel_labels, "dim1": list(IIT_METRICS.keys())}

        return {"PhiID": (PhiID_vals, out_meta), "inf_dyn": (inf_dyn_vals, out_phi), "IIT": (IIT_vals, out_IIT)}



filepath = "/home/yorguin/scratch/data/MEG_ketamine/derivatives/prepDur30Ov20/sub-S041213N1/ses-ketamine/meg/sub-S041213N1_ses-ketamine_task-resting_desc-None_split-01_epo.fif"
meg = read_epochs(filepath, preload=True)
meg = meg.resample(100, npad="auto")  # Resample to 100 Hz

this_epoch = meg.get_data()[0, :, :]  # Get the first epoch data (shape: channels x timepoints)
phi_epoch = process(this_epoch, tau=5, redundancy="MMI", kind="gaussian", channel_labels=meg.ch_names)

meg._data = meg._data[0:1, :, :]  # Keep only the first epoch

phi_epoch2 = single_atoms(meg, tau=5, redundancy="MMI", kind="gaussian", )

phi_epoch3 = atoms_results(phi_epoch2, key='InformationDynamics')

phi_epoch4 = atoms_results(phi_epoch2, key='IntegratedInformationDecomposition')
phi_epoch5 = atoms_results(phi_epoch2, key='IntegratedInformationTheory')

from pprint import pprint
# compare results
orig_infdynam = phi_epoch['inf_dyn'][0]
my_infdynam = phi_epoch3['values'][0,:,:]

pprint(orig_infdynam == my_infdynam)

orig_iit = phi_epoch['IIT'][0]
my_iit = phi_epoch5['values'][0,:,:]

pprint(orig_iit == my_iit)

orig_phi = phi_epoch['PhiID'][0]
my_phi = phi_epoch4['values'][0,:,:]
pprint(orig_phi == my_phi)

