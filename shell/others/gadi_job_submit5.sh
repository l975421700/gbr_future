#!/bin/bash

#PBS -q copyq
#PBS -l walltime=05:00:00
#PBS -l mem=192GB
#PBS -l jobfs=192GB
#PBS -l ncpus=1
#PBS -l storage=gdata/v46
#PBS -l wd
#PBS -P v46

echo "Current time : " $(date +"%T")
cd $HOME
source /home/563/qg8515/miniconda3/bin/activate rcm_gbr
which python

python /home/563/qg8515/code/gbr_future/python/others/py_scripts/qrun8.py

echo "Current time : " $(date +"%T")

