import os
import subprocess
import tqdm


ZA = input('Enter ZA (e.g. 94239): ')
home_directory = input("Enter home directory (here for cwd): ")
if home_directory == "here":
	home_directory = os.getcwd()

mode = input("Enter mode ace/pendf: ")
if mode == "ace":
	mode = '03c'

move_index_lower = int(input('Enter lower index for move: '))
move_index_upper = int(input('Enter upper index for move: '))

target_directory = input('Enter target directory: ')
files = os.listdir(home_directory)

for i in tqdm.tqdm(range(move_index_lower, move_index_upper + 1)):
	subprocess.run(f'mv {home_directory}/{ZA}_{i}.{mode} {target_directory}', shell=True)