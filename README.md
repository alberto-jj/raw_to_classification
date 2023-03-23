# The ``raw_to_classification`` workflow

This repository include all the codes necesary to reproduce the manuscript entitled *"Fully reproducible Python workflow for multi-site resting-state EEG analysis: From raw data to group classification"*

The flowchart below describes the workflow modules.

![diagram_raw_to_classification](https://user-images.githubusercontent.com/71186117/225244708-b0227c35-eef3-42c1-b649-b619e1b41851.png)

The workflow takes raw resting-state EEG recordings in both Brain Imaging Data Structure (BIDS) as well as non-BIDS raw recordings in several formats (.edf, .set, .bdf, .boxy, .vhdr, .cnt, .ctf, etc).

This workflow wraps on multiple tools already available such as [sovaBIDS](https://github.com/yjmantilla/sovabids), [PyPREP](https://github.com/sappelhoff/pyprep), [autoreject](https://github.com/autoreject/autoreject), [mne-icalabel](https://github.com/mne-tools/mne-icalabel), [YASA](https://github.com/raphaelvallat/yasa), [neuroComBat](https://github.com/Jfortin1/ComBatHarmonization), [XGBoost](https://github.com/dmlc/xgboost), and [SHAP](https://github.com/slundberg/shap).


## Installation

```bash
git clone https://github.com/alberto-jj/raw_to_classification.git
cd raw_to_classification
pip install .
```

## Developer Installation

```bash
git clone https://github.com/alberto-jj/raw_to_classification.git
cd raw_to_classification
pip install -e .
```

## Getting started
All configuration parameters can be defined in two core files: `datasets.yml` and `pipeline.yml`.

First, open the `datasets.yml` file and set a name for the dataset(s), then configure parameters listed below:

| Parameter | Input type | Description |
| --- | --- | --- |
| **`url`** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), optional | URL address of the dataset. |
| **`dataset_label`** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required | Identifier label for the dataset. |
| **`participants_file`** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required | Path to the participants metadata file. If data is in BIDS format, fill-in the path to the "participants.tsv" file. |
| **`reader`** | optional |  Fill below the desired <ins>reader function</ins> and its <ins>arguments</ins>. ***Only useful if metadata is NOT in BIDS format*** |
| *`function`* | [`str`](https://docs.python.org/3/library/stdtypes.html#str), optional  | Reader function to be used to read the participants metadata (e.g. pd.read_excel, pd.read_csv).
| *`args`* | [`str`](https://docs.python.org/3/library/stdtypes.html#str), optional  | Fill below the arguments for the reader function (e.g. delimiter: "\\t", or "{}" if no used arguments).
| **`df_transform`** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), optional |  Write any function to organize metadata file (e.g. "df=df.dropna(subset =['id'])"). ***Only useful if metadata is NOT in BIDS format*** |
| **`cleaned_participants`** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), optional  | Path to the resulting participants.tsv after using `reader` or `df_transform` functions.
| **`raw_layout`** | required  | Fill below the arguments for the raw layout.
| *`extension`* | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required  | Extension of the raw files (e.g. ".edf", ".set").
| *`suffix`* | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required  | Suffix for BIDS format (e.g. "eeg").
| *`return_type`* | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required  | Return a list with the desired output (e.g. "filename").
| *`task`* | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required  | Task label according to BIDS specification (e.g. "rest", "eyesClosed").
| **`example_file`** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required  | Path to an exemplary file wether in BIDS or Non-BIDS format.
| **`ch_names`** | [`list`](https://docs.python.org/3/library/stdtypes.html#list), required  | List of Channel names in standard format (i.e. Fp1, Oz).
| **`PowerLineFrequency`** | [`int`](https://docs.python.org/3/library/functions.html#int), required  | Power line noise in Hz.
| **`bids_root`** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required  | Path to the BIDS root folder.
| **`sovabids`** | optional  | Use the `paths` and `rules` parameters to convert into BIDS.
| *`paths`* | | Use the `source_path` and `bids_path` parameters as detailed below.
| `source_path` |[`str`](https://docs.python.org/3/library/stdtypes.html#str)  | Path of the folder with the source files.
| `bids_path` | [`str`](https://docs.python.org/3/library/stdtypes.html#str) | Path of the folder with the BIDS converted data.
| *`rules`* | | Use the `source_path` and `bids_path` parameters as detailed below.
| `dataset_description` | [`str`](https://docs.python.org/3/library/stdtypes.html#str) | Description of the current dataset.
| `Name` |  [`str`](https://docs.python.org/3/library/stdtypes.html#str) | Dataset name.
| `Authors` | [`str`](https://docs.python.org/3/library/stdtypes.html#str)  | Names of the authors of the dataset.
| `sidecar` | | Define below the configurations of the sidecar file.
| `PowerLineFrequency` | [`int`](https://docs.python.org/3/library/functions.html#int)  | Power line noise, noted for visualization and inspection.
| `EEGReference` |  [`str`](https://docs.python.org/3/library/stdtypes.html#str) | Reference channel.
| *`channels`* | | Define below the channels.tsv file. 
| `type` |   | This property allow us to overwrite channel types inferred by MNE 'HEOG', 'VEOG'. Here the syntax is "<channel name> : <channel type according to bids notation>" (e.g. HEOG : HEOG).
| `VEOG` |  [`str`](https://docs.python.org/3/library/stdtypes.html#str) | Vertical EOG channel.
| `HEOG` |  [`str`](https://docs.python.org/3/library/stdtypes.html#str) | Horizontal EOG channel.
| *`non-bids`* |   | Additional configuration not belonging specifically to any of the previous objects
| `eeg_extension` |  | | Sets which extension to read as an eeg file.
| `path_analysis` |  | | Some bids properties can be inferred from the path of the source files.
| `pattern` |  | | TBD
| `operation` |  | | TBD
| `entities.subject` |  | | TBD
| `entities.task` |  | | TBD
| *`file_filter`* |  | | TBD
| `exclude` |  | | TBD




Parameters of the `pipeline.yml` are listed below:



## Running the workflow on publicly-available datasets

Exemplary datasets can be analyzed using our pipeline to reproduce the results of our pre-print "Multi-site band power analysis of resting-state EEG: A tutorial workflow from raw data to group-level classification"

Dataset 1 was collected in 31 subjects (15 Parkinson Disease Patients; and 16 Healthy Controls) at University of California - San Diego. [Available here](
https://openneuro.org/datasets/ds002778/versions/1.0.2)

Besides, Dataset 2 was collected in 38 subjects (19 Parkinson Disease Patients; and 19 Healthy Controls) at the University of Turku, Finland.
[Available here](https://osf.io/pehj9/)


## References

## License


# TODO
- [x] datasets.yml parameters
- [ ] pipeline.yml parameters
- [ ] Add delight to the experience when all tasks are complete :tada:     
