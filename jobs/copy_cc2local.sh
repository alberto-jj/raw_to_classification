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