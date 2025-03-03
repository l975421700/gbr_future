# ~2h

year=$1
# year=2016

module load cdo/2.4.3
shopt -s nullglob

for month in $(seq 1 12); do
    # month=1
    month=$(printf '%02d' $month)
    ymfolder=/scratch/v46/qg8515/data/obs/jaxa/clp/$year$month
    if [ -d $ymfolder ]; then
        echo Processing year $year month $month
        last_day=$(date -d "$year-$month-01 +1 month -1 day" +%d)
        for day in $(seq 1 $last_day); do
            # day=4
            day=$(printf '%02d' $day)
            dfolder=$ymfolder/$day
            if [ -d $dfolder ]; then
                echo Processing year $year month $month day $day
                for hour in $(seq 0 23); do
                    # hour=0
                    hour=$(printf '%02d' $hour)
                    hfolder=$dfolder/$hour
                    if [ -d $hfolder ]; then
                        echo Processing year $year month $month day $day hour $hour
                        for minute in 00 10 20 30 40 50; do
                            # minute=00
                            ofile=$hfolder/CLTYPE_${year}${month}${day}${hour}${minute}.nc
                            if [ -f $ofile ]; then
                                echo Processed year $year month $month day $day hour $hour minute $minute
                            else
                                echo Warning 4: No file for year $year month $month day $day hour $hour minute $minute
                                wget -q -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' --tries=5 --retry-connrefused --wait=5 --timeout=30 --glob=on -A "NC_H*_$year$month${day}_$hour${minute}_L2CLP010_FLDK.02401_02401.nc" ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/$year$month/$day/$hour/ -P /scratch/v46/qg8515/data/obs/jaxa/clp
                                if [ $? -eq 0 ]; then
                                    downloadedfile=$(ls $hfolder/NC_H*_$year$month${day}_$hour${minute}_L2CLP010_FLDK.02401_02401.nc | tail -n 1)
                                    if [ -f $downloadedfile ]; then
                                        cdo -f nc4 -z zip0 -L -setcalendar,gregorian -settaxis,${year}-${month}-${day},${hour}:${minute}:00,1day -selvar,CLTYPE $downloadedfile $ofile
                                        if [ $? -eq 0 ]; then
                                            echo Processed year $year month $month day $day hour $hour minute $minute
                                        else
                                            echo Warning 6: Failed cdo operation
                                        fi
                                    else
                                        echo Warning 5: Cannot find file
                                    fi
                                else
                                    echo No such file
                                fi
                            fi
                        done
                    else
                        echo Warning 3: No folder for year $year month $month day $day hour $hour
                    fi
                done
            else
                echo Warning 2: No folder for year $year month $month day $day
            fi
        done
    else
        echo Warning 1: No folder for year $year month $month
    fi
done

