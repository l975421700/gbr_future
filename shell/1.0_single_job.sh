#PBS -N single_job
#PBS -q hugemem
#PBS -l walltime=0:30:00
#PBS -l mem=1470GB
#PBS -l jobfs=100MB
#PBS -l ncpus=48
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55
#PBS -P v46
#PBS -r y

cd $HOME
source miniconda3/bin/activate lowclouds
python code/gbr_future/shell/0_runpy/run3.py

# code/gbr_future/shell/2.0_get_clean_jaxa.sh 2015 07

#---- remarks
# max walltime in copyq:10hours
# max cpu in copyq:     1
# max J in copyq:       10
# delete all jobs: qstat | awk 'NR > 3 {print $1}' | xargs qdel
# check project quota: lquota
# check user usage: quota
# check queue status: qstat -Q
# https://opus.nci.org.au/spaces/Help/pages/236881198/Queue+Limits...
# nci_account -P v46 -v
# nci_account -P gx60 -v
# nci_account -P gb02 -v
# nci-files-report -S --project v46 --filesystem scratch

# qsub -I -q normal -P v46 -l walltime=48:00:00,ncpus=1,mem=4GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/gx60
# cp -aruvP /home/563/qg8515/scratch/cylc-run1/u-ds714 /home/563/qg8515/scratch/cylc-run/ &

