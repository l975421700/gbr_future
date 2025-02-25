
for idx in $(seq 1 10); do
    qsub -v idx=$idx <<EOF
#PBS -N qjob$idx
#PBS -q normal
#PBS -l walltime=4:00:00
#PBS -l mem=192GB
#PBS -l jobfs=10GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38
#PBS -P v46
#PBS -r y

cd $HOME
source ${HOME}/miniconda3/bin/activate rcm_gbr
python ${HOME}/code/gbr_future/shell/0_runpy/run${idx}.py

# module load python3/3.12.1
# /apps/python3/3.12.1/bin/python3 /home/563/qg8515/code/gbr_future/python/3_obs/3.0_satellites/3.0.0.0_get_CL_Frequency.py -y 2016 -m \${idx}
# /apps/python3/3.12.1/bin/python3 /home/563/qg8515/code/gbr_future/python/3_obs/3.0_satellites/3.0.0.1_get_monthly_CL_Frequency.py -y 2016 -m \${idx}
# /apps/python3/3.12.1/bin/python3 /home/563/qg8515/code/gbr_future/python/3_obs/3.0_satellites/3.0.0.4_get_monthly_CL_RelFre.py -y 2016 -m \${idx}

EOF
done


