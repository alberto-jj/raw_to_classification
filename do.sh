mamba activate sova
python -u 3_preprocess.py > 3_preprocessPOLANDHenry.log
python -u 3b_prepInspection.py
python -u 4_features.py > 4_featuresPOLANDHenry.log
python -u 5_aggregate.py > 5_aggregatePOLANDHenry.log
mamba activate automl
python -u 6_scalingAndFolding.py > 6_scalingAndFoldingPOLANDHenry.log
python -u 6b_aggregateInspection.py > 6b_aggregateInspectionPOLANDHenry.log
