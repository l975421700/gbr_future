
import glob
qjobo_fl = sorted(glob.glob('scratch/others/qjob*'))
for ifile in qjobo_fl:
    with open(ifile, 'r') as file:
        content = file.read()
        if ('Exit Status:        0' in content):
            pass
        # elif not ('Exit Status:        ' in content):
        #     pass
        else:
            print(f'#-------------------------------- Warning')
            print(f'Not success: {ifile}')



