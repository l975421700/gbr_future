
year=$1
month=$2
module load cdo/2.4.3
shopt -s nullglob

for day in $(ls /home/563/qg8515/data/obs/jaxa/clp/${year}${month}); do
  for hour in $(ls /home/563/qg8515/data/obs/jaxa/clp/${year}${month}/${day}); do
    mkdir -p /scratch/v46/qg8515/data/obs/jaxa/clp/${year}${month}/${day}/${hour}
    for minute in 00 10 20 30 40 50; do
        inputfile=$(find /home/563/qg8515/data/obs/jaxa/clp/${year}${month}/${day}/${hour} -type f -name "NC_H??_${year}${month}${day}_${hour}${minute}_L2CLP010_FLDK.02401_02401.nc")
        echo $inputfile

        if [ -z "$inputfile" ]; then
            echo 'NO FILE'
        else
            outputfile=/scratch/v46/qg8515/data/obs/jaxa/clp/${year}${month}/${day}/${hour}/CLP_${year}${month}${day}${hour}${minute}.nc
            echo $outputfile
            if [ -n "$outputfile" ] && [ -f "$outputfile" ]; then
                rm "$outputfile"
            fi

            cdo -f nc4 -z zip0 -L -setcalendar,gregorian -settaxis,${year}-${month}-${day},${hour}:${minute}:00,1day -selvar,CLOT,CLTT,CLTH,CLER_23,CLTYPE,QA "$inputfile" "$outputfile"
        fi
    done
  done
done

# year='2016'
# month='01'
# day='01'
# hour='00'
# minute='00'

# inputfile=/home/563/qg8515/data/obs/jaxa/clp/${year}${month}/${day}/${hour}/NC_H08_${year}${month}${day}_${hour}${minute}_L2CLP010_FLDK.02401_02401.nc
