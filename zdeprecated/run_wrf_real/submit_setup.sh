#PBS -N setupWRF
#PBS -q normal
#PBS -l walltime=04:00:00
#PBS -l mem=16GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46

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
python /home/563/qg8515/code/gbr_future/run_wrf_real/setup_for_wrf.py -c /home/563/qg8515/code/gbr_future/run_wrf_real/config.json

