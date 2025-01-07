
for day in $(seq 1 31); do
    day=$(printf '%02d' $day)
    for month in $(seq 1 12); do
        month=$(printf '%02d' $month)
        qsub -v month=$month,day=$day <<EOF
#PBS -N qjob_${month}_$day
#PBS -q copyq
#PBS -l walltime=3:00:00
#PBS -l mem=6550MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46
#PBS -P v46
#PBS -r y

/home/563/qg8515/code/gbr_future/shell/2.1_get_jaxa.sh 2016 \$month \$day

EOF
    done
done
