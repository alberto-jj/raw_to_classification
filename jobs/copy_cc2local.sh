## run in mobaxterm locally, not logged in to the cluster
rsync -r -h --copy-links --no-perms --progress yorguin@cedar.computecanada.ca:/home/yorguin/scratch/DeepCuriosity/ /media/Y/computecanada/swimmer


rsync -r -h --copy-links --no-perms --progress yorguin@narval.computecanada.ca:/home/yorguin/scratch/saint/aggregate/FeaturesChannels@prep-defaultprep /media/Y/computecanada/swimmer


ssh yorguin@cedar.computecanada.ca     "find /home/yorguin/scratch/data/MEG_*/derivatives/features@prepDur30Ov20/ -type f -name '*.npy'" > files.txt
rsync -h --copy-links --no-perms --progress --files-from=files.txt     --relative yorguin@cedar.computecanada.ca:/ /media/Y/computecanada/cocosprint

ssh yorguin@cedar.computecanada.ca     "find /home/yorguin/scratch/data/MEG_*/derivatives/features@prepDur30Ov20/ -type f -name '*.npy'" > files.txt
rsync -h --copy-links --no-perms --progress --files-from=files.txt     --relative yorguin@cedar.computecanada.ca:/ /media/Y/computecanada/cocosprint

ssh yorguin@cedar.computecanada.ca "find /home/yorguin/scratch/data/MEG_*/derivatives/features@prepDur30Ov20/ -type f -name '*.npy' ! -name '*_PowerSpectrum.npy'" > files.txt


ssh yorguin@cedar.computecanada.ca     "find /home/yorguin/scratch/data/MEG_*/derivatives/prepDur30Ov20" > files.txt



ssh yorguin@cedar.computecanada.ca "find /home/yorguin/scratch/data/MEG_*/meg_data_BIDS/" > files.txt
rsync -h --copy-links --no-perms --progress --dry-run --files-from=files.txt --relative yorguin@cedar.computecanada.ca:/ /home/yorguin/scratch/data/
rsync -h --copy-links --no-perms --progress --dry-run --files-from=files.txt --relative yorguin@cedar.computecanada.ca:/ ~/scratch/data/


find /home/yorguin/scratch/data/MEG_*/meg_data_BIDS/ > bids_files.txt

## To see the total size of all files listed 
xargs -a bids_files.txt du -ch | tail -1

sed 's|/home/yorguin/scratch/data/||' bids_files.txt > rel_bids_files.txt

## https://docs.alliancecan.ca/wiki/Project_layout

mkdir $HOME/projects/def-kjerbi/data_sprint
setfacl -d -m g::rwx $HOME/projects/def-kjerbi/data_sprint
chmod g+s $HOME/projects/def-kjerbi/data_sprint

## this works, not sure why rsync ends up with the files assigned to the personal group (almost no space there)
cat rel_bids_files.txt | xargs -I{} cp --parents "{}" /home/yorguin/projects/def-kjerbi/data_sprint

## This does not work, disk quota exceeded
##cd /home/yorguin/scratch/data
##rsync -av --files-from=rel_bids_files.txt /home/yorguin/scratch/data/ /project/def-kjerbi/data_sprint/
