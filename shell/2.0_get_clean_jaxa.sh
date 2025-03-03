
year=$1
month=$2
# year=2022
# month=12

module load cdo/2.4.3
shopt -s nullglob


# download directories
wget -q -r -nH --cut-dirs=5 --continue --no-remove-listing --no-parent --reject "*.nc" --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/$year$month -P /scratch/v46/qg8515/data/obs/jaxa/clp

ymfolder=/scratch/v46/qg8515/data/obs/jaxa/clp/$year$month

if [ -d $ymfolder ]; then
    echo Processing year month: $year $month
    for day in $(ls $ymfolder); do
        echo Processing day: $day
    # for day in $(seq 13 13); do
    #     day=$(printf '%02d' $day)
    #     echo Processing day: $day
        for hour in $(ls $ymfolder/$day); do
            echo Processing hour: $hour
        # for hour in $(seq 0 4); do
        #     hour=$(printf '%02d' $hour)
        #     echo Processing hour: $hour
            for minute in 00 10 20 30 40 50; do
                echo Processing minute: $minute
                wget -q -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' --glob=on -A "NC_H*_$year$month${day}_$hour${minute}_L2CLP010_FLDK.02401_02401.nc" ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/$year$month/$day/$hour/ -P /scratch/v46/qg8515/data/obs/jaxa/clp
                downloadedfile=$(ls $ymfolder/$day/$hour/NC_H*_$year$month${day}_$hour${minute}_L2CLP010_FLDK.02401_02401.nc | tail -n 1)
                if [ -f $downloadedfile ]; then
                    cleanedfile=$ymfolder/$day/$hour/CLTYPE_${year}${month}${day}${hour}${minute}.nc
                    if [ -n $cleanedfile ] && [ -f $cleanedfile ]; then
                        rm $cleanedfile
                    fi
                    cdo -f nc4 -z zip0 -L -setcalendar,gregorian -settaxis,${year}-${month}-${day},${hour}:${minute}:00,1day -selvar,CLTYPE $downloadedfile $cleanedfile
                    if [ -f $cleanedfile ]; then
                        rm $downloadedfile
                    else
                        echo Warning: no output file
                    fi
                else
                    echo Warning: no file downloaded
                fi
            done
        done
    done
else
    echo Warning: year month $year $month not existing
fi



# wget -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/2016\${month}/* -P /scratch/v46/qg8515/data/obs/jaxa/clp

# wget -r -nH --cut-dirs=5 --continue --no-remove-listing --ftp-user='gaoqg229_gmail.com' --ftp-password='SP+wari8' ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/CLP/010/201601/03/01/NC_H08_20160103_0140_L2CLP010_FLDK.02401_02401.nc -P /home/563/qg8515/data/obs/jaxa/clp

# lftp -u gaoqg229_gmail.com,SP+wari8 ftp.ptree.jaxa.jp
# ls /pub/himawari/L2/CLP/010


