#! /bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --job-name=example
#SBATCH --output=%j-example.out
#SBATCH --error=%j-example.err
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
pip install -r requirements_minimal.txt
pip install .
python -u test_job.py