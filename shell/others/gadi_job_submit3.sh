#!/bin/bash

#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=192GB
#PBS -l ncpus=1
#PBS -l storage=gdata/v46

cd $HOME
source /home/563/qg8515/miniconda3/bin/activate rcm_gbr
python /home/563/qg8515/code/gbr_future/python/others/py_scripts/qrun3.py


