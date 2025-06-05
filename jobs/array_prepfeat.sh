#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-331
## 0-977 for saint
## 0-488 for cocosprint
## you need to get the index range using the command below:
## sbatch --export=STEP=index,PIPELINE_YML=project_files/pipeline_saint.yml --array=0 array_prepfeat.sh
#SBATCH --job-name=prepfeat
#SBATCH --output=%A_%a-prepfeat.out
#SBATCH --error=%A_%a-prepfeat.err
## uncomment if you want to receive emails
##SBATCH --mail-user=yjmantilla@gmail.com
##SBATCH --mail-type=ALL


## Note, to test use:
## sbatch --export=STEP=3,PIPELINE_YML=project_files/pipeline_saint.yml --array=0-2 array_prepfeat.sh
## sbatch --export=STEP=4,PIPELINE_YML=project_files/pipeline_saint.yml --array=0-2 array_prepfeat.sh

## To just run the script, use:
## sbatch --export=STEP=3,PIPELINE_YML=project_files/pipeline_saint.yml array_prepfeat.sh
## sbatch --export=STEP=4,PIPELINE_YML=project_files/pipeline_saint.yml array_prepfeat.sh

## To get the total number of indexes for array job configuration:
## sbatch --export=STEP=index,PIPELINE_YML=project_files/pipeline_saint.yml --array=0 array_prepfeat.sh

## You can use this job as an example of how to run in on an interactive session.
## Remind that you need to have installed the virtual environment before running this script.



ENV=${ENV:-DEFAULT}
if [ "$ENV" == "TMP" ]; then
    ## module load StdEnv/2020
    module load StdEnv/2023
    module load python/3.11.5
    virtualenv --no-download $SLURM_TMPDIR/env
    source $SLURM_TMPDIR/env/bin/activate
    ## pip install --no-index --upgrade pip
    cd /home/yorguin/projects/def-kjerbi/yorguin/raw_to_classification
    pip install --no-index -r requirements_cc_noindex.txt
    pip install -r requirements_extra.txt
    pip install .
elif [ "$ENV" == "DEFAULT" ]; then
    module purge
    ## module load StdEnv/2020
    module load StdEnv/2023
    module load python/3.11.5

    ## The commented lines below should have already been done in the environment setup.
    ## virtualenv --no-download /home/yorguin/envs/raw_to_classification_env
    source /home/yorguin/envs/raw_to_classification_env/bin/activate
    cd /home/yorguin/projects/def-kjerbi/yorguin/raw_to_classification
    ## pip install --no-index -r requirements_cc_noindex.txt
    ## pip install -r requirements_extra.txt
    ## pip install -e .
fi

# Set default YAML path if not passed
PIPELINE_YML=${PIPELINE_YML:-project_files/pipeline_saint.yml}

if [ ! -f "$PIPELINE_YML" ]; then
    echo "Warning: YAML file '$PIPELINE_YML' does not exist!"
fi
## python -u scripts/4_features.py project_files/pipeline_cocosprint.yml --index 0 --retry_errors
## python -u scripts/3_preprocess.py project_files/pipeline_cocosprint.yml --only_total
## python -u scripts/4_features.py project_files/pipeline_cocosprint.yml --retry_errors --only_total
# Decide which step to run based on $STEP
if [ "$STEP" == "3" ]; then
    python -u scripts/3_preprocess.py "$PIPELINE_YML" --index $SLURM_ARRAY_TASK_ID --external_jobs 1 --internal_jobs 1 --retry_errors
elif [ "$STEP" == "4" ]; then
    python -u scripts/4_features.py "$PIPELINE_YML" --index $SLURM_ARRAY_TASK_ID --retry_errors
elif [ "$STEP" == "index" ]; then
    python -u scripts/3_preprocess.py "$PIPELINE_YML" --only_total
else
    echo "Error: Unknown STEP '$STEP'. Use STEP=3 or STEP=4 or STEP=index"
    exit 1
fi
