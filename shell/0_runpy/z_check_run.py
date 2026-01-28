

import glob
import os
qjobo_fl = sorted(glob.glob('/home/563/qg8515/scratch/run/qjob*'))
for ifile in qjobo_fl:
    with open(ifile, 'r') as file:
        content = file.read()
        # if ('Number of Files: ' not in content):
        # if ('Exit Status:        0' not in content):
        #     print(ifile)
        if ('Exit Status:        0' in content):
            os.remove(ifile)
        # else:
        #     print(f'#-------------------------------- Warning')
        #     print(f'Not success: {ifile}')

