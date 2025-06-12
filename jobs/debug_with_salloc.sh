## salloc --time=1:0:0 --mem-per-cpu=16G --ntasks=1 --account=def-kjerbi
module purge
module load StdEnv/2023
module load python/3.11.5
source /home/yorguin/envs/raw_to_classification_env/bin/activate
cd /home/yorguin/scratch/code/raw_to_classification
## or
#cd /home/yorguin/raw_to_classification
ipython

## salloc --time=2:59:0 --mem-per-cpu=16G --ntasks=1 --account=def-kjerbi

tar -cf - /home/yorguin/scratch/data/MEG_psilocybin/bids | xz -T${SLURM_CPUS_PER_TASK} > Datapsibids.tar.xz
tar -cf - /home/yorguin/scratch/data/MEG_psilocybin/derivatives/features@prepDur30Ov20 | xz -T${SLURM_CPUS_PER_TASK} > DatapsiFeats.tar.xz



salloc --time=3:0:0 --mem-per-cpu=16G --ntasks=1 --account=def-kjerbi

salloc --time=5:0:0 --mem-per-cpu=1G   --ntasks=1 --cpus-per-task=32 --account=def-kjerbi

salloc --time=5:0:0 --mem-per-cpu=8G   --ntasks=1 --cpus-per-task=32 --account=def-kjerbi

salloc --time=3:0:0 --mem-per-cpu=64G   --ntasks=1 --cpus-per-task=10 --account=def-kjerbi



