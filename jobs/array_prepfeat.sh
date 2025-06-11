#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --array=110,106,109,107,111,108,114,112,113,117,115,116,167,169,166,168,171,170,174,173,176,175,172,177,119,120,121,122,123,118,129,128,124,125,126,127,74,71,73,75,72,70,76,77,80,78,81,79,144,147,143,146,142,145,150,151,148,149,153,152,63,58,61,60,59,62,67,69,68,64,66,65,34,35,36,38,39,37,42,43,45,40,41,44,205,207,202,204,203,206,208,210,212,209,211,213,179,180,182,183,178,181,184,188,186,189,185,187,156,157,158,154,159,155,161,163,160,165,162,164,50,47,49,48,51,46,56,54,57,53,55,52,191,194,195,190,193,192,200,197,196,201,198,199,131,135,130,132,134,133,136,141,137,138,139,140,83,82,85,87,86,84,91,92,93,89,88,90,26,22,25,23,27,24,31,29,30,32,28,33,95,97,94,99,96,98,103,102,104,101,100,105,16,13,14,15,21,17,19,20,18,217,218,216,219,215,214,221,222,223,225,220,224,326,327,328,329,310,311,322,323,298,299,300,301,318,319,312,313,324,325,316,317,320,321,306,307,330,331,304,305,296,297,302,303,314,315,308,309,231,230,235,234,227,226,247,246,259,258,237,236,253,252,257,256,245,244,233,232,241,240,243,242,255,254,239,238,265,264,263,262,261,260,249,248,229,228,251,250,293,292,269,268,285,284,277,276,273,272,271,270,281,280,275,274,295,294,289,288,279,278,291,290,287,286,267,266,283,282,349,348,361,360,353,352,359,358,333,332,347,346,337,336,343,342,357,356,339,338,345,344,355,354,341,340,351,350,335,334
## 0-977 for saint
## 0-488 for cocosprint
## you need to get the index range using the command below:
## sbatch --export=STEP=index,PIPELINE_YML=project_files/pipeline_saint.yml --array=0 array_prepfeat.sh
## or missing indexes from inspect_only
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
## sbatch --export=STEP=4,PIPELINE_YML=project_files/pipeline_cocosprint.yml array_prepfeat.sh

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
    cd /home/yorguin/scratch/code/raw_to_classification
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
    cd /home/yorguin/scratch/code/raw_to_classification
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
elif [ "$STEP" == "inspect" ]; then
    python -u scripts/4_features.py "$PIPELINE_YML" --inspect_only
else
    echo "Error: Unknown STEP '$STEP'. Use STEP=3 or STEP=4 or STEP=index"
    exit 1
fi
