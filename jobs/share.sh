# List of folders
folders=(
    #/home/yorguin/scratch/data/MEG_ketamine
    #/home/yorguin/scratch/data/MEG_LSDV2
    /home/yorguin/scratch/data/MEG_perampanel
    /home/yorguin/scratch/data/MEG_psilocybin
    /home/yorguin/scratch/data/MEG_tiagabine
)

# Loop through each folder
for folder in "${folders[@]}"; do
    echo "Processing $folder"

    # Prepare group
    echo "Setting group to def-kjerbi for $folder"
    chgrp -Rv def-kjerbi "$folder"

    # Set correct permissions on folders
    echo "Setting permissions for directories in $folder"
    find "$folder" -type d -print -exec chmod g+rx {} \;

    # Set correct permissions on files
    echo "Setting permissions for files in $folder"
    find "$folder" -type f -print -exec chmod g+r {} \;

    # Ensure new files inherit group
    echo "Setting setgid bit for directories in $folder"
    find "$folder" -type d -print -exec chmod g+s {} \;

    echo "Done with $folder"
    echo "-----------------------------"
done
