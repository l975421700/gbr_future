#!/bin/bash

for hour in $(seq 0 23); do
    hour=$(printf '%02d' $hour)
    for month in $(seq 1 12); do
        month=$(printf '%02d' $month)
        qsub -v month=$month,hour=$hour <<EOF
#!/bin/bash
#PBS -N qjob_$month_$hour
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=6550MB
#PBS -l jobfs=6550MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46
#PBS -P v46
#PBS -r y

/home/563/qg8515/code/gbr_future/shell/get_jaxa.sh 2016 \$month \$hour

EOF
    done
done
