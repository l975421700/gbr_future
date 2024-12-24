#!/bin/bash
#PBS -N qjob
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=4GB
#PBS -l jobfs=4GB
#PBS -l ncpus=1
#PBS -J 1-10
#PBS -j oe
#PBS -l storage=gdata/v46
#PBS -P v46
#PBS -r y

cd $HOME
source ${HOME}/miniconda3/bin/activate rcm_gbr
# python ${HOME}/code/gbr_future/python/others/py_scripts/qrun1.py
/home/563/qg8515/code/gbr_future/shell/get_jaxa.sh 2016 2016 1 12 $PBS_ARRAY_INDEX $PBS_ARRAY_INDEX



# max walltime in copyq:10hours
# max cpu in copyq:     1
# max J in copyq:       10
# delete all jobs: qstat | awk 'NR > 2 {print $1}' | xargs qdel
# check project quota: lquota