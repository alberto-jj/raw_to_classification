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
| **url** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), optional | URL address of the dataset. |
| **dataset_label** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required | Identifier label for the dataset. |
| **participants_file** | [`str`](https://docs.python.org/3/library/stdtypes.html#str), required | Path to the participants metadata file. If data is in BIDS format, fill-in the path to the "participants.tsv" file. |
| **reader** | Function | Reader function to be used for reading the participants metadata. Only useful if metadata is NOT in BIDS format. |


- **url:** *(str, optional)* URL address of the dataset.

- **dataset_label:** *(str, required)* Identifier label for the dataset.


- **participants_file:** *(str, required)* Path to the participants metadata file. If data is in BIDS format, fill-in the path to the "participants.tsv" file.

- **reader:** Reader function to be used for reading the participants metadata. ***Only useful if metadata is NOT in BIDS format***
  - **function:**
  - **args:**

- **df_transform:** 

- **cleaned_participants:**

- **raw_layout:**

  - **extension:**

  - **suffix:**

  - **return_type:**

  - **task:**

- **example_file:**

- **ch_names:**

- **PowerLineFrequency:**

- **bids_root:** (str). Path to the BIDS root folder. ***Required field***

- **sovabids:**
  - **paths:**
    - **source_path:**
    - **bids_path:**
  - **rules:**
    - **dataset_description**
    - **Name:**
    - **Authors:**
    - **sidecar:**
      - **PowerLineFrequency:**
      - **EEGReference:**
    - **channels:**
      - **type:**
        - **VEOG:**
        - **HEOG:**
      - **non-bids:**
        - **eeg_extension:**
        - **path_analysis:**
          - **pattern:**
          - **operation:**
            - **entities.subject:**
            - **entities.task:**
        - **file_filter:**
          - **exclude:**




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
