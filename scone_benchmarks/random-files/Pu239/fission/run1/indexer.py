import os
import numpy as np
import subprocess
import tqdm

pendf_dir = '/home/rnt26/uncertaintyanalysis/dataprocessing/random-files/Pu239/fission/run2000_3999/pendf'
# ace_dir = '/home/rnt26/uncertaintyanalysis/dataprocessing/random-files/Pu239/fission/run2000_3999/ace'

# mode = input('ace/pendf: ')

# if mode == 'ace':



pendfs = os.listdir(pendf_dir)
# 	aces = os.listdir(ace_dir)


original_indices = list(range(0,2000))
new_indices = list(range(2000, 4000))

for old_i, new_i in tqdm.tqdm(zip(original_indices, new_indices), total=len(original_indices)):

	# new_ace_filename = f'94239_{new_i}.03c'
	# old_ace_filename = f'94239_{old_i}.03c'


	new_pendf_filename = f'94239_{new_i}.pendf'
	old_pendf_filename = f'94239_{old_i}.pendf'

	# print(f'{old_i} -> {new_i} \n')

	# subprocess.run(f'mv {ace_dir}/{old_ace_filename} {ace_dir}/{new_ace_filename}', shell=True)

	subprocess.run(f'mv {pendf_dir}/{old_pendf_filename} {pendf_dir}/{new_pendf_filename}', shell=True)

