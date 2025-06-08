python -u "./scripts/0_inspect.py" "project_files/pipeline_test.yml"
python -u "./scripts/1_dataset2bids.py" "project_files/pipeline_test.yml"
python -u "./scripts/2_participants.py" "project_files/pipeline_test.yml"
python -u "./scripts/3_preprocess.py" "project_files/pipeline_test.yml" --max_files 1 --external_jobs 1 --raise_on_error
