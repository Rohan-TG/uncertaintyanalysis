import numpy as np
import tqdm
import time
import subprocess
import datetime
import os
import sys
sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')
from groupEnergies import Pu239, Reactions

start_time = time.time()
ZA = Pu239.ZA # ZA for Pu-240
group = 15
ACE_file_directory = f'/home/rnt26/uncertaintyanalysis/data/Pu239/fission/g15/run1'
scone_executable_path = '/home/rnt26/scone/SCONE/Build/scone.out'

num_cores = int(input('Num. cores: ')) # number of cores to use for this specific instance of scone

perturbation_coefficients = np.arange(-0.500, 0.000, 0.001)
perturbation_coefficients = [round(i,3) for i in perturbation_coefficients]

default_xsfile = 'endfb-viii0.xsfile'
current_dir = os.getcwd()
target_dir = 'scone_benchmarks'
idx = current_dir.find(target_dir)
default_Jezebel = 'Jezebel'

target_path = current_dir[:(idx + len(target_dir))]
libfilename = f'lib{ZA}.xsfile'
mt = Reactions.fission

subprocess.run(f"cp {target_path}/{default_xsfile} {libfilename}", shell=True)
subprocess.run(f"cp {target_path}/{default_Jezebel} Jezebel", shell=True)

search_files = os.listdir()
for file in search_files:
	if '.xsfile' in file:
		libfile = file
		break

xsfile_fullpath = os.path.abspath(libfile)
os.environ["SCONE_ACE"] = xsfile_fullpath

if 'outputfiles' not in search_files:
	subprocess.run('mkdir outputfiles', shell=True)

for coefficient in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):
	input_coefficient = round(coefficient, 3) # Coefficient string prep
	ACE_filename = f"{ACE_file_directory}/Pu239_g{group}_{input_coefficient:0.3f}_MT{mt}.09c" # name of ACE file for this SCONE run

	with open(libfile, 'r') as file:
		lines = file.readlines()

	try:
		for i, ACE_address in enumerate(lines):
			if str(ZA) in ACE_address:
				Nuclide_ACE_address = i
				break
	except:
		print('Cannot find ACE address match. Terminating')
		break

	lines[Nuclide_ACE_address] = f'{ZA}.00c; 1; {ACE_filename};\n' # Rewrite line

	with open(libfile, 'w') as file: # write to new .xsfile
		file.writelines(lines)

	subprocess.run(f'{scone_executable_path} --omp {num_cores} Jezebel', shell=True) # run scone
	subprocess.run(f'mv output.m outputfiles/output-{input_coefficient:0.3f}.m', shell=True) # move output file to output directory for later analysis

end_time = time.time()

print(f"Time elapsed: {datetime.timedelta(seconds=end_time-start_time)}")
