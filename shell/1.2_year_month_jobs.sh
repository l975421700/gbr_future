for year in $(seq 2017 2024); do
    for month in $(seq 1 12); do
        # month=$(printf '%02d' $month)
        echo $year $month
        qsub -v year=$year,month=$month <<EOF
#PBS -N qjob2_${year}$(printf '%02d' $month)
#PBS -q hugemem
#PBS -l walltime=5:00:00
#PBS -l mem=245GB
#PBS -l jobfs=100MB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l storage=gdata/v46+scratch/v46+gdata/rr1+gdata/rt52+gdata/ob53+gdata/oi10+gdata/hh5+gdata/fs38+gdata/zv2+gdata/ra22+scratch/public+gdata/qx55+gdata/gx60+gdata/py18+gdata/rv74+gdata/xp65
#PBS -P if69
#PBS -r y

cd ${HOME}

# source miniconda3/bin/activate lowclouds
# python code/gbr_future/shell/0_runpy/run2.py -y $year -m $month
# code/gbr_future/shell/2.0_get_clean_jaxa.sh $year $month

module use /g/data/xp65/public/modules
module load conda/analysis3
/g/data/xp65/public/apps/med_conda_scripts/analysis3-25.10.d/bin/python code/gbr_future/shell/0_runpy/run2.py -y $year -m $month

EOF
    done
done

