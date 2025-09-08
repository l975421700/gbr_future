

import glob
import os
qjobo_fl = sorted(glob.glob('scratch/others/qjob*'))
for ifile in qjobo_fl:
    with open(ifile, 'r') as file:
        content = file.read()
        if ('Exit Status:        0' in content):
            # print(ifile)
            os.remove(ifile)
        else:
            print(f'#-------------------------------- Warning')
            print(f'Not success: {ifile}')

