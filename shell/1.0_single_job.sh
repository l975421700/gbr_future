#PBS -N qjob
#PBS -q hugemem
#PBS -l walltime=1:00:00
#PBS -l mem=1470GB
#PBS -l jobfs=100MB
#PBS -l ncpus=96
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/py18+gdata/gx60
#PBS -P v46
#PBS -r y

cd $HOME
source miniconda3/bin/activate lowclouds
python code/gbr_future/shell/0_runpy/run5.py

# code/gbr_future/shell/2.0_get_clean_jaxa.sh 2015 07

#---- remarks
# max walltime/cpu/J in copyq:10hours/1/10
# delete all jobs: qstat | awk 'NR > 3 {print $1}' | xargs qdel
# check project quota: lquota
# check user usage: quota
# check queue status: qstat -Q
# https://opus.nci.org.au/spaces/Help/pages/236881198/Queue+Limits...
# nci_account -P v46 -v # ACCESS:gx60,lg87;Center:gb02,ng72,if69,fy29;BoM:hd50
# nci-files-report -S --project gx60 --filesystem scratch

# qsub -I -q normal -P v46 -l walltime=48:00:00,ncpus=1,mem=4GB,storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+scratch/public+gdata/zv2+gdata/ra22+gdata/gx60
# cp -aruvP /home/563/qg8515/gdata_v46/data/sim/um/BARRA-C2-RAL3.3 /home/563/qg8515/data/sim/um/ &

