mamba activate sova
python -u 0_inspect.py pipeline_r2c.yml > 0_inspect_r2c.log
python -u 1_dataset2bids.py pipeline_r2c.yml > 1_dataset2bids_r2c.log
python -u 2_participants.py pipeline_r2c.yml > 2_participants_r2c.log
python -u 3_preprocess.py pipeline_r2c.yml --external_jobs 10 > 3_preprocess_r2c.log
python -u 3b_prepInspection.py pipeline_r2c.yml > 3b_prepInspection_r2c.log
python -u 4_features.py pipeline_r2c.yml --external_jobs 10 > 4_features_r2c.log
python -u 5_aggregate.py pipeline_r2c.yml > 5_aggregate_r2c.log
python -u 5.5_aggregateInspection.py pipeline_r2c.yml > 5.5_aggregateInspection_r2c.log
mamba activate automl
python -u 6_scalingAndFolding.py > 6_scalingAndFoldingPOLANDHenry.log
python -u 6b_aggregateInspection.py > 6b_aggregateInspectionPOLANDHenry.log

python -u 3_preprocess.py pipeline_saint.yml --external_jobs 1 --only_total
python -u 4_features.py pipeline_saint.yml --only_total