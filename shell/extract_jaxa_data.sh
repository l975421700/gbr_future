
for month in $(seq 1 12); do
    month=$(printf '%02d' $month)
    for day in $(seq 1 31); do
        day=$(printf '%02d' $day)
            qsub -v month=$month,day=$day <<EOF
#PBS -N qjob_${month}_${day}
#PBS -q normal
#PBS -l walltime=16:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46
#PBS -P v46
#PBS -r y

cd $HOME

/home/563/qg8515/code/gbr_future/shell/extract_jaxa_data/extract_CLTYPE.sh 2016 \$month \$day

# module load python3/3.12.1
# python3 /home/563/qg8515/code/gbr_future/python/3_obs/3.0_satellites/3.0.0_get_himawari_data.py -y 2016 -m \$month -d \$day

EOF
    done
done
