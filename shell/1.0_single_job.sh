#PBS -N single_job
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l mem=192GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38
#PBS -P v46
#PBS -r y

# cp -ruv /scratch/v46/qg8515/data/obs/jaxa/clp /home/563/qg8515/data/obs/jaxa

# /home/563/qg8515/code/gbr_future/shell/2.0_get_clean_jaxa.sh 2015 07

cd $HOME
source ${HOME}/miniconda3/bin/activate rcm_gbr
python /home/563/qg8515/code/gbr_future/shell/0_runpy/run1.py


# #PBS -J 1-10
# /home/563/qg8515/code/gbr_future/shell/get_jaxa.sh 2016 2016 1 12 $PBS_ARRAY_INDEX $PBS_ARRAY_INDEX

#---- remarks
# #PBS -q hugemem
# max walltime in copyq:10hours
# max cpu in copyq:     1
# max J in copyq:       10
# delete all jobs: qstat | awk 'NR > 2 {print $1}' | xargs qdel
# check project quota: lquota
# #PBS -l jobfs=4GB
# https://opus.nci.org.au/spaces/Help/pages/236881198/Queue+Limits...
# lquota: check project usage
# quota: check user usage
# qstat -Q: check queue status