# The ``raw_to_classification`` workflow

This repository include all the codes necesary to reproduce the manuscript entitled *"Multi-site band power analysis of resting-state EEG: A tutorial workflow from raw data to group-level classification"*

The flowchart below describes the workflow modules.

<img src="https://user-images.githubusercontent.com/71186117/225244708-b0227c35-eef3-42c1-b649-b619e1b41851.png" width="420" height="500">

The workflow takes raw resting-state EEG recordings in both Brain Imaging Data Structure (BIDS) as well as non-BIDS raw recordings in several formats (.edf, .set, .bdf, .vhdr, .cnt, .ctf, etc).

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
---

## Getting started
All configuration parameters can be defined in two core files: `datasets.yml` and `pipeline.yml`.

1. Open the `datasets.yml` file and set a name for the dataset(s), then configure its parameters.

<details>
    <summary><b> Click to see the parameters for the <code style="font-family: consolas;">`datasets.yml`</code></b> </summary>
    <p>


| Parameter | Input type | Description |
| --- | --- | --- |
| **`url`** | `str`, optional | URL address of the dataset. |
| **`dataset_label`** | `str`	, required | Identifier label for the dataset. |
| **`participants_file`** | `str`, required | Path to the participants metadata file. If data is in BIDS format, fill-in the path to the "participants.tsv" file. |
| **`reader`** | optional |  Fill below the desired <ins>reader function</ins> and its <ins>arguments</ins>. ***Only useful if metadata is NOT in BIDS format*** |
| *`function`* | `str`, optional  | Reader function to be used to read the participants metadata (e.g. pd.read_excel, pd.read_csv).
| *`args`* | `str`, optional  | Fill below the arguments for the reader function (e.g. delimiter: "\\t", or "{}" if no used arguments).
| **`df_transform`** | `str`, optional |  Write any function to organize metadata file (e.g. "df=df.dropna(subset =['id'])"). ***Only useful if metadata is NOT in BIDS format*** |
| **`cleaned_participants`** | `str`	, optional  | Path to the resulting participants.tsv after using `reader` or `df_transform` functions.
| **`raw_layout`** | required  | Fill below the arguments for the raw layout.
| *`extension`* | `str`	, required  | Extension of the raw files (e.g. ".edf", ".set").
| *`suffix`* | `str`	, required  | Suffix for BIDS format (e.g. "eeg").
| *`return_type`* | `str`	, required  | Return a list with the desired output (e.g. "filename").
| *`task`* | `str`	, required  | Task label according to BIDS specification (e.g. "rest", "eyesClosed").
| **`example_file`** | `str`	, required  | Path to an exemplary file wether in BIDS or Non-BIDS format.
| **`ch_names`** | `list`	, required  | List of Channel names in standard format (i.e. Fp1, Oz).
| **`PowerLineFrequency`** | `int`	, required  | Power line noise in Hz.
| **`bids_root`** | `str`	, required  | Path to the BIDS root folder.
| **`sovabids`** | optional  | Use the `paths` and `rules` parameters to convert into BIDS.
| *`paths`* | | Use the `source_path` and `bids_path` parameters as detailed below.
| `source_path` |`str`	  | Path of the folder with the source files.
| `bids_path` | `str`	 | Path of the folder with the BIDS converted data.
| *`rules`* | | Use the `source_path` and `bids_path` parameters as detailed below.
| `dataset_description` | `str`	 | Description of the current dataset.
| `Name` |  `str`	 | Dataset name.
| `Authors` | `str`	  | Names of the authors of the dataset.
| `sidecar` | | Define below the configurations of the sidecar file.
| `PowerLineFrequency` | `int`	  | Power line noise, noted for visualization and inspection.
| `EEGReference` |  `str`	 | Reference channel.
| *`channels`* | | Define below the channels.tsv file. 
| `type` |   | This property allow us to overwrite channel types inferred by MNE 'HEOG', 'VEOG'. Here the syntax is "<channel name> : <channel type according to bids notation>" (e.g. HEOG : HEOG).
| `VEOG` |  `str`	 | Vertical EOG channel.
| `HEOG` |  `str`	 | Horizontal EOG channel.
| *`non-bids`* |   | Additional configuration not belonging specifically to any of the previous objects
| `eeg_extension` |  |  Sets which extension to read the EEG files.
| `path_analysis` |  | Some BIDS properties can be inferred from the path of the source files.
| `pattern` |  | Regex pattern of the original EEG filenames (e.g. data/%a%_%b%.set if the names follows the "subject" + "task" + ".set" pattern, as in 01_rest.set. See [`sovabids`](https://sovabids.readthedocs.io/en/latest/rules_schema.html) documentation for more details. 
| `operation` |  |  Make an operation between fields extracted by pattern matching to produce a single BIDS field (e.g. Given "Healthy_01_EyesOpen.set", one can produce "Healthy01" in the "subject" BIDS field by using the operation *entities.subject : "[a] + [b]"* See [ `sovabids` ](https://sovabids.readthedocs.io/en/latest/rules_schema.html#operation-experimental) documentation for more details. 
| `entities.subject` |  |  Pattern element in the original filename corresponding to the subject identifier (e.g. Given original filenames like "sub01_rest.set", "sub02_rest.set", "subXX_rest.set", the pattern %a%_%b%.set can be used. Thus, [a] should be used in the `entities.subject` field to extract the participant identifier from filenames.
| `entities.task` |  |  Pattern element in the original filename corresponding to the task name (e.g. Given original filenames like "sub01_rest.set", "sub02_rest.set", "subXX_rest.set", the pattern %a%_%b%.set can be used. Thus, [b] should be used in the `entities.subject` field to extract the task name ("rest") from filenames.
| *`file_filter`* |  |  Fill below with the parameters to filter and select files.
| `exclude` |  |   Substring present in the filenames to exclude.


</p>
</details>


2. Open the pipeline.yml file and define its parameters.
      
<details>
    <summary><b> Click here to see the parameters for the <code style="font-family: consolas;">`pipeline.yml`</code></b> </summary>
    <p>


| Parameter | Input type | Description |
| --- | --- | --- |
| **`inspect`** |  | Inspection module to visualize stacked all channels PSD plots. |
| *`path`* | `str`| Fill in the path to the inspection plots and log results |
| **`preprocess`** |  | Preprocessing module. |
| *`prepare`* |  |  Fill below the desired parameters for the PREPARE pipeline |
| *`epoch_length`* | `int` | Epoch length in seconds.  |
| *`downsample`* | `int` | New sampling frequency after downsampling all datasets to a common sample frequency. Default value is 500 |
| *`ica_method`* | `str` | Methods for ICA fitting available in [`mne.preprocessing.ICA`](https://mne.tools/stable/generated/mne.preprocessing.ICA.html). Default is `fastica` |
| *`skip_prep`* | `bool` | Skip `pyprep`. Default value is False. |
| *`skip_reject`* | `int` | Skip `autoreject`. Default value is False.  |
| *`overwrite`* | `bool` | Overwrite original file. Default value is False. |
| **`features`** |  | Feature extraction module. |
| *`downsample`* |  `int` |  New sampling frequency after downsampling all datasets to a common sample frequency. Default value is 500. **Useful if the preprocessing module was skipped** |
| *`num_epochs`* | `int`, `str`| Equalize epoch number across subjects to a fixed numeric value, or the min of epochs across all subjects.  |
| *`prefilter`* |  | Prefilter signals based on a defined range of interest frequencies|
| `l_freq` | `int` | The lower pass-band edge, FIR filter[`mne.Epochs.filter`](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.filter). |
| `h_freq` | `int` | The higher pass-band edge, FIR filter[`mne.Epochs.filter`](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.filter).|
| *`keep_channels`* | `bool` or `list`  | List with the selected channels to keep, if False, all channels are used for feature extraction |
| *`feature_list`* | `bool` | You can configure different features for the same function but different args |
| *`PowerSpectrum`* |  | Compute power spectrum using using multitapers (https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.filter). |
| `h_freq` | `int` | The higher pass-band edge, FIR filter[`mne.Epochs.filter`](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.filter).|
| *`keep_channels`* | `bool` or `list`  | List with the selected channels to keep, if False, all channels are used for feature extraction |
| *`feature_list`* | `bool` | You can configure different features for the same function but different args |

    PowerSpectrum:
      overwrite : False
      function : spectrum
      args:
        multitaper:
          adaptive : False
          low_bias : True
          normalization : 'full'
          verbose : 0
    RelativeBandPower1:
      overwrite : False
      function: relative_bandpower
      args :
        bands:
          delta : [1,4]
          theta : [4,8]
          alpha : [8,13]
          beta  : [13,30]
          pre_alpha : [5.5,8]
          slow_theta : [4,5.5]
        multitaper : {}
    RelativeBandPower2:
      overwrite : False
      function: relative_bandpower
      args :
        bands:
          alpha1 : [8.5, 10.5]
          alpha2 : [10.5, 12.5]
          beta1  : [12.5, 18.5]
          beta2  : [18.5, 21]
          beta3  : [21, 30]
        multitaper : {}

        aggregate:
          path : './data/aggregate'
          filename : 'multidataset.csv'
          id_splitter : '/'
          features: 
            - RelativeBandPower1
            #- RelativeBandPower2

        harmonization:
          path : './data/harmonization'
          MAX_FEATURES: null
          neuroCombat:
            batch : dataset
            covars :
              - dataset
              - sex
              - age
              - group
            drop :
              - id
              - subject
              - task
            categorical:
              - sex
              - group
              - dataset # Cuando entra en neurocombat el batch (Dataset) se quita de la lista
          split:
            test : 0.3
            target : 'group'
            folds : 5

        classification:
          path: './data/classification'

</p>
</details>

3. Start running the `raw_to_classification` pipeline `.py` files from 0 to 7.

---

## Running the workflow on publicly-available datasets

Exemplary datasets can be analyzed using our pipeline to reproduce the results of our pre-print "Multi-site band power analysis of resting-state EEG: A tutorial workflow from raw data to group-level classification"

Dataset 1 was collected in 31 subjects (15 Parkinson Disease Patients; and 16 Healthy Controls) at University of California - San Diego. [Available here](
https://openneuro.org/datasets/ds002778/versions/1.0.2)

Besides, Dataset 2 was collected in 38 subjects (19 Parkinson Disease Patients; and 19 Healthy Controls) at the University of Turku, Finland.
[Available here](https://osf.io/pehj9/)

---

## References

## License

## TODO

- [x] datasets.yml parameters
- [ ] pipeline.yml parameters
- [ ] add nested-cross validation (shufflesplit CV) [here the paper ](https://www.nature.com/articles/s41598-022-23327-1) with the .py code 
- [ ] Implement automljar as default for the ML module :tada:
- [ ] per epoch feature
- [ ] add aggregation per roi
- [ ] find fold N for nvars stratify
