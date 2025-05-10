#! /bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-977
##0-977
#SBATCH --job-name=prepfeat
#SBATCH --output=%A_%a-prepfeat.out
#SBATCH --error=%A_%a-prepfeat.err
## uncomment if you want to receive emails
##SBATCH --mail-user=yjmantilla@gmail.com
##SBATCH --mail-type=ALL

module purge
##module load StdEnv/2020
module load StdEnv/2023
module load python/3.11.5

cd /home/yorguin/envs

##virtualenv --no-download $SLURM_TMPDIR/env
##virtualenv --no-download raw_to_classification_env

##source $SLURM_TMPDIR/env/bin/activate
source raw_to_classification_env/bin/activate

##pip install --no-index --upgrade pip
cd /home/yorguin/raw_to_classification

## pip install --no-index -r requirements_minimal.txt
## pip install -r requirements_extra.txt
## pip install -e .
##python -u 3_preprocess.py pipeline_saint.yml --index $SLURM_ARRAY_TASK_ID --external_jobs 1 --internal_jobs 1 --retry_errors
python -u 4_features.py pipeline_saint.yml --index $SLURM_ARRAY_TASK_ID --retry_errors


