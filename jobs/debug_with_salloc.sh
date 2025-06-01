## salloc --time=1:0:0 --mem-per-cpu=16G --ntasks=1 --account=def-kjerbi
module load StdEnv/2023
module load python/3.11.5
source /home/yorguin/envs/raw_to_classification_env/bin/activate
cd /home/yorguin/projects/def-kjerbi/yorguin/raw_to_classification
## or
#cd /home/yorguin/raw_to_classification
ipython