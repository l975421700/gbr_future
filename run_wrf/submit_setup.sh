#!/bin/bash
#PBS -N setupWRF
#PBS -q normal
#PBS -l walltime=00:30:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -j oe
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
ulimit -s unlimited

source ${HOME}/miniconda3/bin/activate rcm_gbr
python setup_for_wrf.py -c config.json

