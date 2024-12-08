#!/bin/bash
#PBS -N compNcSht
#PBS -q normal
#PBS -l walltime=2:00:00
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -j oe
#PBS -l wd
#PBS -l storage=gdata/${PROJECT}

module load netcdf/4.7.1

tempfile=`mktemp -p . -u`
tempfilemgc=${tempfile}.mgc

cat <<EOF > $tempfilemgc
# NetCDF-V3
0  string   CDF\001    NetCDF Data Format data
# NetCDF-V3-64bit
0  string   CDF\002    NetCDF Data Format data, 64-bit offset

EOF

file -C -m $tempfilemgc

## find files              check type        netcdf only                    extract the filename           convert to netcdf4 and compress fields
find $1 -type f -exec file -m $tempfile {} \; | grep "NetCDF Data Format data" | rev | cut -d: -f2- | rev | xargs -I@ bash -c "echo @ ; nccopy -d1 @ @_nc4 ; mv @_nc4 @"

rm -f $tempfilemgc $tempfile

echo "## End of script ##"

