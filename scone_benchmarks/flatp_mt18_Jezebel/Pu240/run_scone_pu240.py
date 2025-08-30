import numpy as np
import tqdm
import time
import subprocess
import datetime
import os

start_time = time.time()

ACE_file_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/data/Pu240/fmt18' # location of the ECCO Group 6 perturbed ACE files to be used for generating samples

scone_executable_path = '/home/rnt26/scone/SCONE/Build/scone.out' # location of the scone executable


perturbation_coefficients = np.arange(-0.500, 0.501, 0.001)

ZA = 94240

default_xsfile = 'endfb-viii0.xsfile'
current_dir = os.getcwd()
target_dir = 'scone_benchmarks'
idx = current_dir.find(target_dir)
default_Jezebel = 'Jezebel'

target_path = current_dir[:(idx + len(target_dir))]
libfilename = f'lib{ZA}.xsfile'


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
	# subprocess.run('echo $SCONE_ACE', shell=True)

	input_coefficient = round(coefficient, 3) # Coefficient string prep

	ACE_filename = f"{ACE_file_directory}/Flat_Pu-240_{input_coefficient:0.3f}_MT18.09c" # name of ACE file for this SCONE run

	with open(libfile, 'r') as file:
		lines = file.readlines()

	for i, line in enumerate(lines): # Find location of the nuclide in the .xsfile
		if str(ZA) in line:
			nuclide_ACE_address = i
			break

	lines[nuclide_ACE_address] = f'{ZA}.00c; 1; {ACE_filename};\n' # Rewrite line

	with open(libfile, 'w') as file: # write to new lib1.xsfile
		file.writelines(lines)


	num_cores = int(input('Core no.: '))# number of cores to use for this specific instance of scone
	subprocess.run(f'{scone_executable_path} --omp {num_cores} Jezebel', shell=True) # run scone


	subprocess.run(f'mv output.m outputfiles/output-{input_coefficient:0.3f}.m', shell=True) # move output file to output directory for later analysis



end_time = time.time()

print(f"Time elapsed: {datetime.timedelta(seconds=end_time-start_time)}")
