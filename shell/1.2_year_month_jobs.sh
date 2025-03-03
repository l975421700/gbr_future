
for year in $(seq 2015 2024); do
    for month in $(seq 1 12); do
        # month=$(printf '%02d' $month)
        echo $year $month
        qsub -v year=$year,month=$month <<EOF
#PBS -N qjob_${year}_${month}
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l mem=18GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38
#PBS -P v46
#PBS -r y

module load python3/3.12.1
/apps/python3/3.12.1/bin/python3 /home/563/qg8515/code/gbr_future/python/3_obs/3.0_satellites/3.0.0.0_get_himawari_CLP.py -y $year -m $month

# /home/563/qg8515/code/gbr_future/shell/2.0_get_clean_jaxa.sh $year $month

EOF
    done
done

