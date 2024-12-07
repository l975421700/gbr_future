#!/bin/bash
#PBS -N setupWRF
#PBS -l walltime=10:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -q copyq
#PBS -l wd
#PBS -l storage=gdata/v46

module purge
module load pbs
module load dot
module load intel-compiler/2019.3.199
module load openmpi/4.0.2
module load ncl/6.6.2
module load hdf5/1.10.5
module load netcdf/4.7.1
module load nco
module load wgrib2

source ${HOME}/miniconda3/bin/activate rcm_gbr
which python

ulimit -s unlimited

python ${HOME}/code/gbr_future/run_wrf/setup_for_wrf.py -c config.json

