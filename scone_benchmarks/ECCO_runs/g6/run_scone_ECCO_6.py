import numpy as np
import tqdm
import time
import subprocess
import datetime

start_time = time.time()

ACE_file_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/data/ecco6Pu9/' # location of the ECCO Group 6 perturbed ACE files to be used for generating samples

scone_executable_path = '/home/rnt26/scone/SCONE/Build/scone.out' # location of the scone executable

num_cores = 15 # number of cores to use for this specific instance of scone
# perturbation_coefficients = np.arange(-0.500, 0.501, 0.001)


perturbation_coefficients = [-0.499, 0.000, 0.500]

for coefficient in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):
	libfile = 'lib6.xsfile'

	input_coefficient = round(coefficient, 3) # Coefficient string prep

	ACE_filename = f"{ACE_file_directory}/ECCO33-g6_Pu9_{input_coefficient:0.3f}_MT18.09c" # name of ACE file for this SCONE run

	with open(libfile, 'r') as file:
		lines = file.readlines()

	Pu_239_address = 511 # Line number of the Pu-239 ACE file address

	lines[Pu_239_address] = f'94239.00c; 1; {ACE_filename};\n' # Rewrite line

	with open(libfile, 'w') as file: # write to new lib1.xsfile
		file.writelines(lines)

	subprocess.run(f'{scone_executable_path} --omp {num_cores} Jezebel', shell=True) # run scone

	subprocess.run(f'mv output.m outputfiles/output-{input_coefficient:0.3f}.m', shell=True) # move output file to output directory for later analysis



end_time = time.time()

print(f"Time elapsed: {datetime.timedelta(seconds=end_time-start_time)}")