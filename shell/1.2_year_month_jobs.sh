for year in $(seq 2023 2023); do
    for month in $(seq 12 12); do
        # month=$(printf '%02d' $month)
        echo $year $month
        qsub -v year=$year,month=$month <<EOF
#PBS -N qjob_${year}_$(printf '%02d' $month)
#PBS -q copyq
#PBS -l walltime=10:00:00
#PBS -l mem=4GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22+scratch/public+gdata/qx55+gdata/gx60
#PBS -P v46
#PBS -r y

cd ${HOME}
# source miniconda3/bin/activate lowclouds
# python code/gbr_future/shell/0_runpy/run12.py -y $year -m $month
code/gbr_future/shell/2.0_get_clean_jaxa.sh $year $month

EOF
    done
done

