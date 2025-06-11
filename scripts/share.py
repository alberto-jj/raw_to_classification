import os
import subprocess
# THIS DOES NOT WOR IN ITS CURRENT STATE
def share_path_with_group(path, group="def-kjerbi", execute=True):
    """
    For a given file path, generate and execute chgrp and chmod commands
    to make the path traversable and the file group-readable.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")


    print(f"\nPreparing path for group '{group}': {path}\n")

    # Build list of parent directories (from root down to containing folder)
    parts = path.split(os.sep)
    scratch_idx = parts.index('scratch')  # Ensure 'scratch' is in the path
    parts = parts[scratch_idx:]  # Start from 'scratch' directory
    parents = []
    for i in range(2, len(parts)):  # start at 2 instead of 1
        breakpoint()
        parent_path = os.sep + os.path.join(*parts[1:i])
        if parent_path and os.path.isdir(parent_path):
            parents.append(parent_path)
    print(parents)
    parent_dir = os.path.dirname(path)

    # Ensure parent directories and the containing folder are included
    if parent_dir not in parents:
        parents.append(parent_dir)

    # 1. chgrp and chmod g+rx on all parent folders
    for folder in parents:
        print(f"chgrp {group} {folder}")
        print(f"chmod g+rx {folder}")
        if execute:
            subprocess.run(["chgrp", group, folder], check=True)
            subprocess.run(["chmod", "g+rx", folder], check=True)

    # 2. chgrp and chmod g+r on the file (if it is a file)
    if os.path.isfile(path):
        print(f"chgrp {group} {path}")
        print(f"chmod g+r {path}")
        if execute:
            subprocess.run(["chgrp", group, path], check=True)
            subprocess.run(["chmod", "g+r", path], check=True)
    else:
        print(f"Warning: The path '{path}' is not a file. Skipping file chmod step.")

    print("\nDone.\n")

import glob

root_path = os.path.expanduser('/home/yorguin/scratch/code/raw_to_classification/data')

all_paths = glob.glob(os.path.join(root_path, '**', '*'), recursive=True)

for path in all_paths:
    if os.path.isfile(path) and not os.path.islink(path):
        try:
            share_path_with_group(path, group="def-kjerbi", execute=False)
        except Exception as e:
            print(f"Error processing path '{path}': {e}")
        break
# Note: This script will traverse all directories and files under the specified root path