#!/bin/bash
#PBS -N qjob
#PBS -q copyq
#PBS -l walltime=02:00:00
#PBS -l mem=100GB
#PBS -l jobfs=100GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46

cd $HOME
source ${HOME}/miniconda3/bin/activate rcm_gbr
python ${HOME}/code/gbr_future/python/others/py_scripts/qrun1.py


