import os
import sys

import pandas as pd

computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Pu239, Pu240, Pu241, Ga69, Ga71
import time
import tqdm
import subprocess

scone_executable_path = '/home/rnt26/scone/SCONE/Build/scone.out'

starttime = time.time()

Pu239_ZA = Pu239.ZA
Pu240_ZA = Pu240.ZA
Pu241_ZA = Pu241.ZA
Ga69_ZA = Ga69.ZA
Ga71_ZA = Ga71.ZA

num_cores = int(input('Num. cores: '))

default_xsfile = 'endfb-viii0.xsfile'
current_dir = os.getcwd()
target_dir = 'scone_benchmarks'
idxt = current_dir.find(target_dir)
default_Jezebel = 'Jezebel'

target_path = current_dir[:(idxt + len(target_dir))]
libfilename = f'allobservables.xsfile'

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



Pu239_ACE_file_directory = input('Enter Pu239 ACE file directory: ')
Pu239_ACE_files = os.listdir(Pu239_ACE_file_directory)

Pu240_ACE_file_directory = input('Enter Pu240 ACE file directory: ')
Pu240_ACE_files = os.listdir(Pu240_ACE_file_directory)

Pu241_ACE_file_directory = input('Enter Pu241 ACE file directory: ')
Pu241_ACE_files = os.listdir(Pu241_ACE_file_directory)

Ga69_ACE_file_directory = input('Enter Ga69 ACE file directory: ')
Ga69_ACE_files = os.listdir(Ga69_ACE_file_directory)

Ga71_ACE_file_directory = input('Enter Ga71 ACE file directory: ')
Ga71_ACE_files = os.listdir(Ga71_ACE_file_directory)

used_pu240_files = []
used_pu241_files = []
used_pu239_files = []
used_ga69_files = []
used_ga71_files = []

for pu239_ace_file, pu240_ace_file, pu241_ace_file, ga69_ace_file, ga71_ace_file in tqdm.tqdm(zip(Pu239_ACE_files, Pu240_ACE_files, Pu241_ACE_files, Ga69_ACE_files, Ga71_ACE_files),
																							  total=len(Pu239_ACE_files)):
	pu239_file_index = int(pu239_ace_file.split('.03c')[0].split('_')[1])
	pu240_file_index = int(pu240_ace_file.split('.03c')[0].split('_')[1])
	pu241_file_index = int(pu241_ace_file.split('.03c')[0].split('_')[1])
	ga69_file_index = int(ga69_ace_file.split('.03c')[0].split('_')[1])
	ga71_file_index = int(ga71_ace_file.split('.03c')[0].split('_')[1])

	Pu239_ACE_filename = f'{Pu239_ACE_file_directory}/94239_{pu239_file_index}.03c'

	Pu240_ACE_filename = f'{Pu240_ACE_file_directory}/94240_{pu240_file_index}.03c'

	Pu241_ACE_filename = f'{Pu241_ACE_file_directory}/94241_{pu241_file_index}.03c'

	Ga69_ACE_filename = f'{Ga69_ACE_file_directory}/3169_{ga69_file_index}.03c'

	Ga71_ACE_filename = f'{Ga71_ACE_file_directory}/3171_{ga71_file_index}.03c'

	with open(libfile, 'r') as file: # the ACE address file (.xsfile)
		lines = file.readlines()

	try:
		for i9, ACE_address in enumerate(lines):
			if str(Pu239_ZA) in ACE_address:
				Pu239_xsfile_address = i9
				break

		for i0, ACE_address in enumerate(lines):
			if str(Pu240_ZA) in ACE_address:
				Pu240_xsfile_address = i0
				break

		for i1, ACE_address in enumerate(lines):
			if str(Pu241_ZA) in ACE_address:
				Pu241_xsfile_address = i1
				break

		for i69, ACE_address in enumerate(lines):
			if str(Ga69_ZA) in ACE_address:
				Ga69_xsfile_address = i69
				break

		for i71, ACE_address in enumerate(lines):
			if str(Ga71_ZA) in ACE_address:
				Ga71_xsfile_address = i71
				break
	except:
		print('Cannot find ACE address match. Terminating')
		break

	lines[Pu239_xsfile_address] = f'{Pu239_ZA}.00c; 1; {Pu239_ACE_filename};\n'  # Rewrite line

	lines[Pu240_xsfile_address] = f'{Pu240_ZA}.00c; 1; {Pu240_ACE_filename};\n'

	lines[Pu241_xsfile_address] = f'{Pu241_ZA}.00c; 1; {Pu241_ACE_filename};\n'

	lines[Ga69_xsfile_address] = f'{Ga69_ZA}.00c; 1; {Ga69_ACE_filename};\n'

	lines[Ga71_xsfile_address] = f'{Ga71_ZA}.00c; 1; {Ga71_ACE_filename};\n'

	with open(libfile, 'w') as file:  # write to new .xsfile
		file.writelines(lines)


	used_pu239_files.append(pu239_file_index)
	used_pu240_files.append(pu240_file_index)
	used_pu241_files.append(pu241_file_index)
	used_ga69_files.append(ga69_file_index)
	used_ga71_files.append(ga71_file_index)

	subprocess.run(f'{scone_executable_path} --omp {num_cores} Jezebel', shell=True)  # run scone

	subprocess.run(f'mv output.m outputfiles/output-Pu239_{pu239_file_index}_Pu240_{pu240_file_index}_Pu241_{pu241_file_index}_Ga69_{ga69_file_index}_Ga71_{ga71_file_index}.m',
				   shell=True)  # move output file to output directory for later analysis



index_dataframe = pd.DataFrame({'pu239': used_pu239_files,
								'pu240': used_pu240_files,
								'pu241': used_pu241_files,
								'ga69': used_ga69_files,
								'ga71': used_ga71_files,})

index_dataframe.to_csv('All_nuclides_scone_run_indices.csv')