import os
import subprocess
import tqdm


ZA = input('Enter the ZA combination (e.g. 94239): ')


original_lower_index = int(input('Original lower index: '))
original_upper_index = int(input('Original upper index: '))


original_indices = list(range(original_lower_index,original_upper_index))

new_lower_index = int(input('Enter new lower index: '))
new_upper_index = int(input('Enter new upper index: '))

new_indices = list(range(new_lower_index, new_upper_index))


output_directory = input("Enter output directory (type here for cwd): ")
if output_directory == "here":
	output_directory = os.getcwd()

output_files = os.listdir(output_directory)
for old_i, new_i in tqdm.tqdm(zip(original_indices, new_indices), total=len(original_indices)):
	new_output_filename = f'output-{new_i}.m'
	old_output_filename = f'output-{old_i}.m'

	subprocess.run(f'mv {output_directory}/{old_output_filename} {output_directory}/{new_output_filename}',
				   shell=True)