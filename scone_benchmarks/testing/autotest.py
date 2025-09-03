import os
import tqdm
import time
import datetime
import subprocess
import numpy as np

start_time = time.time()

scone_executable_path = '/home/rnt26/scone/SCONE/Build/scone.out'

num_cores = int(input('Core no.: '))# number of cores to use for this specific instance of scone


files = os.listdir()

# ZA = 94239 # variable
ZA = 31069
MT = 2

num_runs_per_nuclide = 5

default_xsfile = 'endfb-viii0.xsfile'
current_dir = os.getcwd()
target_dir = 'scone_benchmarks'
idx = current_dir.find(target_dir)
default_Jezebel = 'Jezebel'

target_path = current_dir[:(idx + len(target_dir))]

subprocess.run(f"cp {target_path}/{default_Jezebel} Jezebel", shell=True)

ACE_files = []
for f in files:
	if '.09c' in f:
		ACE_files.append(f)


keff_list = [[] for i in range(len(ACE_files))]
keff_err_list = [[] for i in range(len(ACE_files))]
for j, ACE in tqdm.tqdm(enumerate(ACE_files), total=len(ACE_files)):
	for i in tqdm.tqdm(range(num_runs_per_nuclide)):

		libfilename = f'test{ZA}-MT{MT}.xsfile'
		subprocess.run(f"cp {target_path}/{default_xsfile} {libfilename}", shell=True)

		search_files = os.listdir()
		for file in search_files:
			if '.xsfile' in file:
				libfile = file
				break

		xsfile_fullpath = os.path.abspath(libfile)
		os.environ["SCONE_ACE"] = xsfile_fullpath

		ACE_filename = f"{current_dir}/{ACE}"

		with open(libfile, 'r') as file:
			lines = file.readlines()

		for i, line in enumerate(lines): # Find location of the nuclide in the .xsfile
			if str(ZA) in line:
				nuclide_ACE_address = i
				break

		lines[nuclide_ACE_address] = f'{ZA}.00c; 1; {ACE_filename};\n'  # Rewrite line

		with open(libfile, 'w') as file: # write to new lib1.xsfile
			file.writelines(lines)

		subprocess.run(f'{scone_executable_path} --omp {num_cores} Jezebel', shell=True)  # run scone

		with open('output.m', 'r') as outputfile:
			output_lines = outputfile.readlines()

		data = output_lines[12]
		data.replace('E', 'e')
		data = data[15:38]
		data = data.split(',')
		keff = float(data[0])
		keff_err = float(data[1])

		keff_list[j].append(keff)
		keff_err_list[j].append(keff_err)


for i, result in enumerate(keff_list):
	mean = np.mean(result)
	std = np.std(result)

	print(f"For {ACE_files[i]}, k_eff: {mean:0.5f} +- {std:0.5f}")

end_time = time.time()

print(f"Time elapsed: {datetime.timedelta(seconds=end_time-start_time)}")