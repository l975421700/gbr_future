#PBS -N qjob
#PBS -q express
#PBS -l walltime=00:30:00
#PBS -l mem=192GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22
#PBS -P v46
#PBS -r y

cd $HOME
# module load python3/3.12.1
# /apps/python3/3.12.1/bin/python3 ${HOME}/code/gbr_future/shell/0_runpy/run3.py

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
