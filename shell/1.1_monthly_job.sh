
for month in $(seq 1 12); do
    qsub -v month=$month <<EOF
#PBS -N qjob_$month
#PBS -q normal
#PBS -l walltime=02:00:00
#PBS -l mem=32GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46
#PBS -P v46

module load python3/3.12.1
# /apps/python3/3.12.1/bin/python3 /home/563/qg8515/code/gbr_future/python/3_obs/3.0_satellites/3.0.0.0_get_CL_Frequency.py -y 2016 -m \${month}
# /apps/python3/3.12.1/bin/python3 /home/563/qg8515/code/gbr_future/python/3_obs/3.0_satellites/3.0.0.1_get_monthly_CL_Frequency.py -y 2016 -m \${month}
/apps/python3/3.12.1/bin/python3 /home/563/qg8515/code/gbr_future/python/3_obs/3.0_satellites/3.0.0.4_get_monthly_CL_RelFre.py -y 2016 -m \${month}

EOF
done


