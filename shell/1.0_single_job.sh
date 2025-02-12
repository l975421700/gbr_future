#PBS -N single_job
#PBS -q normal
#PBS -l walltime=3:00:00
#PBS -l mem=192GB
#PBS -l jobfs=60GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38
#PBS -P v46
#PBS -r y

# cp -ruv /scratch/v46/qg8515/data/obs/jaxa/clp /home/563/qg8515/data/obs/jaxa

cd $HOME
source ${HOME}/miniconda3/bin/activate rcm_gbr
python ${HOME}/code/gbr_future/shell/0_runpy/run2.py


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
