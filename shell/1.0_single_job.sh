#PBS -N qjob
#PBS -q normal
#PBS -l walltime=1:00:00
#PBS -l mem=96GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/qx55
#PBS -P nf33
#PBS -r y

cd $HOME
source ${HOME}/miniconda3/bin/activate lowclouds
python ${HOME}/code/gbr_future/shell/0_runpy/run1.py

# ${HOME}/code/gbr_future/shell/2.0_get_clean_jaxa.sh 2015 07

#---- remarks
# max walltime in copyq:10hours
# max cpu in copyq:     1
# max J in copyq:       10
# delete all jobs: qstat | awk 'NR > 2 {print $1}' | xargs qdel
# check project quota: lquota
# check user usage: quota
# check queue status: qstat -Q
# https://opus.nci.org.au/spaces/Help/pages/236881198/Queue+Limits...
# nci_account -P v46 -v
# nci_account -P gx60 -v
# nci_account -P gb02 -v

# cp -aruvP /home/563/qg8515/scratch/data /g/data/gx60/qg8515/ &
