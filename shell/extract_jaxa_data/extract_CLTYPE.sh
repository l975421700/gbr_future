
# qsub -I -q normal -l walltime=04:00:00,ncpus=4,mem=4GB,jobfs=4GB,storage=gdata/v46

year=$1
month=$2
day=$3
# year='2016'
# month='01'
# day='01'
# hour='02'
# minute='00'

module load cdo/2.4.3
which cdo
cdo='/apps/cdo/2.4.3/bin/cdo'
shopt -s nullglob

for hour in {00..23}; do
    for minute in 00 10 20 30 40 50; do
        echo ${year} ${month} ${day} ${hour} ${minute}

        inputfile=$(find /home/563/qg8515/data/obs/jaxa/clp/${year}${month}/${day}/${hour} -type f -name "NC_H??_${year}${month}${day}_${hour}${minute}_L2CLP010_FLDK.02401_02401.nc")
        echo $inputfile

        if [ -z "$inputfile" ]; then
            echo 'NO FILE'
        else
            outputfile=/home/563/qg8515/data/obs/jaxa/clp/${year}${month}/${day}/${hour}/CLTYPE_${year}${month}${day}${hour}${minute}.nc
            echo $outputfile
            if [ -n "$outputfile" ] && [ -f "$outputfile" ]; then
                rm "$outputfile"
            fi

            ${cdo} -f nc4 -z zip1 -L -setcalendar,gregorian -settaxis,${year}-${month}-${day},${hour}:${minute}:00,1day -selvar,CLTYPE "$inputfile" "$outputfile"
        fi
    done
done


