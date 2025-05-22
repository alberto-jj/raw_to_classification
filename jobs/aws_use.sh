module load StdEnv/2020
module load gcc/9.3.0
module load fsl/6.0.4

## California
## aws s3 sync --no-sign-request --dryrun s3://openneuro.org/ds002778 /home/yorguin/scratch/datasets/ds002778

## Greece
## aws s3 sync --no-sign-request --dryrun s3://openneuro.org/ds004504 /home/yorguin/scratch/datasets/ds004504

## Oslo
## aws s3 sync --no-sign-request --dryrun s3://openneuro.org/ds003775 /home/yorguin/scratch/datasets/ds003775 --exclude "derivatives/*"

## Poland
## aws s3 sync --no-sign-request --dryrun s3://openneuro.org/ds004796 /home/yorguin/scratch/datasets/ds004796 --exclude "derivatives/*"

## Finland
## rsync -r -h --copy-links --no-perms --progress --exclude 'derivatives/' --dry-run /media/Y/datasets/HenryRailo/bids yorguin@narval.computecanada.ca:/home/yorguin/scratch/datasets/HenryRailo/

## Iowa

## rsync -r -h --copy-links --no-perms --progress --exclude 'derivatives/' --dry-run /media/Y/datasets/Iowa/Dataset/IowaDataset/bids yorguin@narval.computecanada.ca:/home/yorguin/scratch/datasets/iowa/