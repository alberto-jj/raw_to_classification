# Share MEG_ketamine file directory with hamza97 (read only)
setfacl -d -m u:hamza97:rX /home/yorguin/scratch/data/MEG_ketamine/meg_data_BIDS/sub-S021013N51/ses-ketamine/meg
setfacl -R -m u:hamza97:rX /home/yorguin/scratch/data/MEG_ketamine/meg_data_BIDS/sub-S021013N51/ses-ketamine/meg

# Share MEG_LSDV2 file directory with hamza97 (read only)
setfacl -d -m u:hamza97:rX /home/yorguin/scratch/data/MEG_LSDV2/meg_data_BIDS/sub-S3LR/ses-lsd/meg
setfacl -R -m u:hamza97:rX /home/yorguin/scratch/data/MEG_LSDV2/meg_data_BIDS/sub-S3LR/ses-lsd/meg

# Share MEG_tiagabine file directory with hamza97 (read only)
setfacl -d -m u:hamza97:rX /home/yorguin/scratch/data/MEG_tiagabine/meg_data_BIDS/sub-S130810N2/ses-placebo/meg
setfacl -R -m u:hamza97:rX /home/yorguin/scratch/data/MEG_tiagabine/meg_data_BIDS/sub-S130810N2/ses-placebo/meg

# Share MEG_psilocybin file directory with hamza97 (read only)
setfacl -d -m u:hamza97:rX /home/yorguin/scratch/data/MEG_psilocybin/bids/sub-S021211N51/ses-placebo/meg
setfacl -R -m u:hamza97:rX /home/yorguin/scratch/data/MEG_psilocybin/bids/sub-S021211N51/ses-placebo/meg

# Share MEG_perampanel file directory with hamza97 (read only)
setfacl -d -m u:hamza97:rX /home/yorguin/scratch/data/MEG_perampanel/meg_data_BIDS/sub-S041012N2/ses-perampanel/meg
setfacl -R -m u:hamza97:rX /home/yorguin/scratch/data/MEG_perampanel/meg_data_BIDS/sub-S041012N2/ses-perampanel/meg

chmod g+rx /home/yorguin
chmod g+rx /scratch/yorguin
chmod g+rx /scratch/yorguin/data
chmod g+rx /scratch/yorguin/data/MEG_ketamine
chmod g+rx /scratch/yorguin/data/MEG_ketamine/meg_data_BIDS
chmod g+rx /scratch/yorguin/data/MEG_ketamine/meg_data_BIDS/sub-S021013N51
chmod g+rx /scratch/yorguin/data/MEG_ketamine/meg_data_BIDS/sub-S021013N51/ses-ketamine


chgrp -R def-kjerbi /scratch/yorguin/data/MEG_ketamine/meg_data_BIDS/sub-S021013N51
chmod -R g+rx /scratch/yorguin/data/MEG_ketamine/meg_data_BIDS/sub-S021013N51

chgrp def-kjerbi /scratch/yorguin
chmod g+rx /scratch/yorguin

# Fix MEG_ketamine
chgrp -R def-kjerbi /scratch/yorguin/data/MEG_ketamine
chmod -R g+rx /scratch/yorguin/data/MEG_ketamine

# Optional but recommended: propagate def-kjerbi as default group on MEG_ketamine
chmod g+s /scratch/yorguin/data/MEG_ketamine


chgrp def-kjerbi /scratch/yorguin/data/MEG_ketamine
chmod g+rx /scratch/yorguin/data/MEG_ketamine

namei -l /scratch/yorguin/data/MEG_ketamine/meg_data_BIDS/sub-S021013N51/ses-ketamine/meg

ls -l /scratch/yorguin/data/MEG_ketamine/meg_data_BIDS/sub-S021013N51/ses-ketamine/meg


##/home/yorguin/scratch/data/MEG_psilocybin/bids/sub-S020311N50/ses-psilocybin/meg/sub-S020311N50_ses-psilocybin_task-resting_meg.fif


# Set group on all parent folders and file
chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin
chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin/bids
chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin/bids/sub-S020311N50
chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin/bids/sub-S020311N50/ses-psilocybin
chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin/bids/sub-S020311N50/ses-psilocybin/meg
chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin/bids/sub-S020311N50/ses-psilocybin/meg/sub-S020311N50_ses-psilocybin_task-resting_meg.fif

# Add group read+execute on folders
chmod g+rx /scratch/yorguin/data/MEG_psilocybin
chmod g+rx /scratch/yorguin/data/MEG_psilocybin/bids
chmod g+rx /scratch/yorguin/data/MEG_psilocybin/bids/sub-S020311N50
chmod g+rx /scratch/yorguin/data/MEG_psilocybin/bids/sub-S020311N50/ses-psilocybin
chmod g+rx /scratch/yorguin/data/MEG_psilocybin/bids/sub-S020311N50/ses-psilocybin/meg

# Add group read on the file
chmod g+r /scratch/yorguin/data/MEG_psilocybin/bids/sub-S020311N50/ses-psilocybin/meg/sub-S020311N50_ses-psilocybin_task-resting_meg.fif

scp yorguin@cedar.computecanada.ca:/home/yorguin/scratch/data/MEG_psilocybin/bids/sub-S020311N50/ses-psilocybin/meg/sub-S020311N50_ses-psilocybin_task-resting_meg.fif



# # Make /scratch/yorguin accessible
# chgrp def-kjerbi /scratch/yorguin
# chmod g+rx /scratch/yorguin

# # Make /scratch/yorguin/data accessible
# chgrp def-kjerbi /scratch/yorguin/data
# chmod g+rx /scratch/yorguin/data

# chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin
# chmod g+rx /scratch/yorguin/data/MEG_psilocybin


# chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin/bids
# chmod g+rx /scratch/yorguin/data/MEG_psilocybin/bids


# chgrp def-kjerbi /scratch/yorguin/data/MEG_psilocybin/derivatives
# chmod g+rx /scratch/yorguin/data/MEG_psilocybin/derivatives


# chmod g+rx /home/yorguin/scratch/data/MEG_psilocybin/derivatives/features@prepDur30Ov20
# chgrp def-kjerbi /home/yorguin/scratch/data/MEG_psilocybin/derivatives/features@prepDur30Ov20

# chgrp -Rv def-kjerbi /home/yorguin/scratch/data/MEG_psilocybin/derivatives/features@prepDur30Ov20
# chmod -Rv g+rx /home/yorguin/scratch/data/MEG_psilocybin/derivatives/features@prepDur30Ov20


# chgrp -Rv def-kjerbi /home/yorguin/scratch/data/MEG_psilocybin/
# chmod -Rv g+rx /home/yorguin/scratch/data/MEG_psilocybin/

# chgrp -Rv def-kjerbi /home/yorguin/scratch/data/MEG_LSDV2/
# chmod -Rv g+rx /home/yorguin/scratch/data/MEG_LSDV2/

# chmod g+s /home/yorguin/scratch/data/MEG_LSDV2/
# find /home/yorguin/scratch/data/MEG_LSDV2/ -type d -exec chmod g+s {} \;





# chgrp -Rv def-kjerbi /home/yorguin/scratch/data/$DATASET/derivatives/features@prepDur30Ov20
# chmod -Rv g+rx /home/yorguin/scratch/data/$DATASET/derivatives/features@prepDur30Ov20



# # Set group recursively on the folder
# chgrp -R def-kjerbi /scratch/yorguin/data/MEG_psilocybin


# /home/yorguin/scratch/data/MEG_ketamine
# /home/yorguin/scratch/data/MEG_LSDV2
# /home/yorguin/scratch/data/MEG_perampanel
# /home/yorguin/scratch/data/MEG_psilocybin
# /home/yorguin/scratch/data/MEG_tiagabine

# # Prepare group
# chgrp -Rv def-kjerbi /home/yorguin/scratch/data/MEG_LSDV2/

# # Set correct permissions
# find /home/yorguin/scratch/data/MEG_LSDV2/ -type d -exec chmod g+rx {} \;
# find /home/yorguin/scratch/data/MEG_LSDV2/ -type f -exec chmod g+r {} \;

# # Ensure new files will inherit group correctly
# find /home/yorguin/scratch/data/MEG_LSDV2/ -type d -exec chmod g+s {} \;
