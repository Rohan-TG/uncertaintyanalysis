import os
import numpy as np
import subprocess
import tqdm

pendf_dir = input('Enter PENDF dir: ')
ace_dir = input('Enter ACE dir: ')

pendfs = os.listdir(pendf_dir)
aces = os.listdir(ace_dir)



original_indices = list(range(0,2000))

new_lower_index = int(input('Enter new lower index: '))
new_upper_index = int(input('Enter new upper index: '))

new_indices = list(range(new_lower_index, new_upper_index))

for old_i, new_i in tqdm.tqdm(zip(original_indices, new_indices), total=len(original_indices)):

	new_ace_filename = f'94239_{new_i}.03c'
	old_ace_filename = f'94239_{old_i}.03c'


	new_pendf_filename = f'94239_{new_i}.pendf'
	old_pendf_filename = f'94239_{old_i}.pendf'

	# print(f'{old_i} -> {new_i} \n')

	subprocess.run(f'mv {ace_dir}/{old_ace_filename} {ace_dir}/{new_ace_filename}', shell=True)

	subprocess.run(f'mv {pendf_dir}/{old_pendf_filename} {pendf_dir}/{new_pendf_filename}', shell=True)


