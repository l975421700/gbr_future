for year in $(seq 2024 2024); do
    for month in $(seq 1 12); do
        # month=$(printf '%02d' $month)
        echo $year $month
        qsub -v year=$year,month=$month <<EOF
#PBS -N qjob_${year}_$(printf '%02d' $month)
#PBS -q hugemem
#PBS -l walltime=5:00:00
#PBS -l mem=400GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22+scratch/public+gdata/qx55
#PBS -P gx60
#PBS -r y

cd ${HOME}
source miniconda3/bin/activate lowclouds
python code/gbr_future/shell/0_runpy/run4.py -y $year -m $month
# code/gbr_future/shell/2.0_get_clean_jaxa.sh $year $month

EOF
    done
done

