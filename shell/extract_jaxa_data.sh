
for month in $(seq 1 12); do
    month=$(printf '%02d' $month)
    qsub -v month=$month <<EOF
#PBS -N qjob_${month}
#PBS -q normal
#PBS -l walltime=16:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46
#PBS -P v46
#PBS -r y

/home/563/qg8515/code/gbr_future/shell/extract_jaxa_data/extract_CLTYPE.sh 2016 \$month

EOF
done
