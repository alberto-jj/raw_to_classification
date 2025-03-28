#! /bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:10:00
#SBATCH --job-name=example
#SBATCH --output=%j-example.out
#SBATCH --error=%j-example.err
#SBATCH --mail-user=yjmantilla@gmail.com
#SBATCH --mail-type=ALL

## module load python/3.10.2
## cd /home/yorguin/scratch/code
## /home/yorguin/envs/cocoenv/bin/python -u pall_test.py

## os 2.384185791015625e-06
## numpy 8.344650268554688e-06
## logging 7.152557373046875e-06
## datetime 2.4318695068359375e-05
## ruamel 5.366790533065796
## pprint 9.775161743164062e-06
## pandas 82.4442687034607
## glob 6.4373016357421875e-06
## mne 93.09295988082886
## sio 24.44395112991333
## mneconnectvity 153.34460258483887
## matplotlib 88.63852953910828
## h5py 8.748783349990845
## timemask 5.4836273193359375e-06



module load python/3.10.2
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
cd /home/yorguin/projects/def-kjerbi/yorguin/raw_to_classification
##pip install --no-index -r requirements.txt
pip install -r requirements.txt
python -u pall_test.py

## os 3.0994415283203125e-06
## numpy 8.344650268554688e-06
## logging 5.245208740234375e-06
## datetime 1.9073486328125e-05
## ruamel 0.016777753829956055
## pprint 8.106231689453125e-06
## pandas 0.22880768775939941
## glob 1.430511474609375e-06
## mne 0.4260425567626953
## sio 0.04334282875061035
## mneconnectvity 0.7295374870300293
## matplotlib 0.8654437065124512
## h5py 0.03133845329284668
## timemask 4.0531158447265625e-06