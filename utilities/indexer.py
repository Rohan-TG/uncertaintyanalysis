import os
import numpy as np
import subprocess
import tqdm

print('WARNING: Run in correct directory')

ZA = input('Enter the ZA combination (e.g. 94239): ')


original_lower_index = int(input('Original lower index: '))
original_upper_index = int(input('Original upper index: '))


original_indices = list(range(original_lower_index,original_upper_index))

new_lower_index = int(input('Enter new lower index: '))
new_upper_index = int(input('Enter new upper index: '))

new_indices = list(range(new_lower_index, new_upper_index))

mode = input('Enter mode ace/pendf: ')

if mode == 'ace':
	ace_dir = os.getcwd()
	aces = os.listdir(ace_dir)
	for old_i, new_i in tqdm.tqdm(zip(original_indices, new_indices), total=len(original_indices)):

		new_ace_filename = f'{ZA}_{new_i}.03c'
		old_ace_filename = f'{ZA}_{old_i}.03c'

		subprocess.run(f'mv {ace_dir}/{old_ace_filename} {ace_dir}/{new_ace_filename}', shell=True)


if mode == 'pendf':
	pendf_dir = os.getcwd()
	pendfs = os.listdir(pendf_dir)
	for old_i, new_i in tqdm.tqdm(zip(original_indices, new_indices), total=len(original_indices)):

		new_pendf_filename = f'{ZA}_{new_i}.pendf'
		old_pendf_filename = f'{ZA}_{old_i}.pendf'

		subprocess.run(f'mv {pendf_dir}/{old_pendf_filename} {pendf_dir}/{new_pendf_filename}', shell=True)


if mode == 'fix':
	pendf_dir = os.getcwd()
	pendfs = os.listdir(pendf_dir)
	for old_i, new_i in tqdm.tqdm(zip(original_indices, new_indices), total=len(original_indices)):
		new_pendf_filename = f'{ZA}_{new_i}.pendf.xz'
		old_pendf_filename = f'{ZA}_{old_i}.pendf.xz'

		subprocess.run(f'mv {pendf_dir}/{old_pendf_filename} {pendf_dir}/{new_pendf_filename}', shell=True)

