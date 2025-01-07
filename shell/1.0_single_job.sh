#PBS -N qjob
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=4GB
#PBS -l jobfs=4GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46
#PBS -P v46
#PBS -r y


cp -ruv /home/563/qg8515/data/obs/jaxa/clp /scratch/v46/qg8515/data/obs/jaxa/


# cd $HOME
# source ${HOME}/miniconda3/bin/activate rcm_gbr
# python ${HOME}/code/gbr_future/python/others/py_scripts/qrun1.py
# #PBS -J 1-10
# /home/563/qg8515/code/gbr_future/shell/get_jaxa.sh 2016 2016 1 12 $PBS_ARRAY_INDEX $PBS_ARRAY_INDEX

#---- remarks
# max walltime in copyq:10hours
# max cpu in copyq:     1
# max J in copyq:       10
# delete all jobs: qstat | awk 'NR > 2 {print $1}' | xargs qdel
# check project quota: lquota