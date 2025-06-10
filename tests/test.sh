python -u "./scripts/0_inspect.py" "project_files/pipeline_test.yml"
python -u "./scripts/1_dataset2bids.py" "project_files/pipeline_test.yml"
python -u "./scripts/2_participants.py" "project_files/pipeline_test.yml"
python -u "./scripts/3_preprocess.py" "project_files/pipeline_test.yml" --external_jobs 1
python -u "./scripts/3b_prepInspection.py" "project_files/pipeline_test.yml"


python -u "./scripts/0_inspect.py" "project_files/pipeline_cocosprint.yml"
python -u "./scripts/1_dataset2bids.py" "project_files/pipeline_cocosprint.yml"
python -u "./scripts/4_features.py" "project_files/pipeline_cocosprint.yml" --inspect_only
python -u "./scripts/4_features.py" "project_files/pipeline_cocosprint.yml" --inspect_only --raise_on_error

python -u "./scripts/4_features.py" "project_files/pipeline_cocosprint.yml" --inspect_only > inspect_features.log