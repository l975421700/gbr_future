
for year in $(seq 2020 2020); do
    for month in $(seq 1 12); do
        # month=$(printf '%02d' $month)
        echo $year $month
        qsub -v year=$year,month=$month <<EOF
#PBS -N qjob_${year}_${month}
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l mem=192GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22
#PBS -P v46
#PBS -r y

cd ${HOME}
source ${HOME}/miniconda3/bin/activate lowclouds
python ${HOME}/code/gbr_future/shell/0_runpy/run4.py -y $year -m $month

# module load python3/3.12.1
# /apps/python3/3.12.1/bin/python3 ${HOME}/code/gbr_future/python/3_obs/3.0_satellites/3.0.0.0_get_himawari_CLP.py -y $year -m $month

# ${HOME}/code/gbr_future/shell/2.0_get_clean_jaxa.sh $year $month

EOF
    done
done

