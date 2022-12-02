# raw_to_classification
This repository include all the codes necesary to reproduce the manuscript entitled *"Fully reproducible Python workflow for multi-site resting-state EEG analysis: From raw data to group classification"*


The flowchart below describes the workflow


![image](https://user-images.githubusercontent.com/71186117/205314830-4417ea5f-5c19-433b-b2f8-772aca88312d.png)

As input datasets we included resting-state EEG recordings in both Brain Imaging Data Structure (BIDS) as well as non-BIDS raw recordings in .EDF format.

Dataset 1 was collected in 31 subjects (15 Parkinson Disease Patients; and 16 Healthy Controls) at University of California - San Diego.
https://openneuro.org/datasets/ds002778/versions/1.0.2

Besides, Dataset 2 was collected in 38 subjects (19 Parkinson Disease Patients; and 19 Healthy Controls) at the University of Turku, Finland.
https://osf.io/pehj9/

This workflow wraps on multiple tools already available such as [sovaBIDS](https://github.com/yjmantilla/sovabids), [PyPREP](https://github.com/sappelhoff/pyprep), [autoreject](https://github.com/autoreject/autoreject), [mne-icalabel](https://github.com/mne-tools/mne-icalabel), [YASA](https://github.com/raphaelvallat/yasa), [neuroComBat](https://github.com/Jfortin1/ComBatHarmonization), [XGBoost](https://github.com/dmlc/xgboost), and [SHAP](https://github.com/slundberg/shap).
