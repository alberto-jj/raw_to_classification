#! /bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-1
##0-250
#SBATCH --job-name=preprocess
#SBATCH --output=%A_%a-preprocess.out
#SBATCH --error=%A_%a-preprocess.err
## uncomment if you want to receive emails
##SBATCH --mail-user=yjmantilla@gmail.com
##SBATCH --mail-type=ALL

module load StdEnv/2020
module load StdEnv/2023
module load python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

##pip install --no-index --upgrade pip
cd /home/yorguin/projects/def-kjerbi/yorguin/raw_to_classification
##pip install --no-index -r requirements.txt
pip install --no-index -r requirements_minimal.txt
pip install -r requirements_extra.txt
pip install .
python -u 3_preprocess.py pipeline_saint.yml --index $SLURM_ARRAY_TASK_ID --external_jobs 1 --internal_jobs 1


