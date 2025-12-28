import os
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
from groupEnergies import Reactions
import tqdm
import ENDF6
import numpy as np

processes = int(input("Num. processes: "))
outputs_directory = input("Enter SCONE outputs directory: ")
output_files = os.listdir(outputs_directory)

lower_energy_bound = float(input("Lower energy bound: "))


pu239_pendf_directory = input("Pu-239 PENDF directory: ")
pu240_pendf_directory = input("Pu-240 PENDF directory: ")
pu241_pendf_directory = input("Pu-241 PENDF directory: ")

parquet_directory = os.getcwd()



keff_list = []
keff_error_list = []

file_index_list = []

print("Reading output files...")

for outputfile in tqdm.tqdm(output_files, total=len(output_files)):
	read_object = open(f'{outputs_directory}/{outputfile}', 'r')

	Pu239_file_index = int(outputfile.split('.m')[0].split('_')[1])

	Pu240_file_index = int(outputfile.split('.m')[0].split('_')[3])

	Pu241_file_index = int(outputfile.split('.m')[0].split('_')[5])

	lines = read_object.readlines()
	keffline = lines[12]

	keff_value_string = keffline[15:26]
	keff_value_float = float(keff_value_string.replace('E', 'e'))
	keff_list.append(keff_value_float)

	keff_error = keffline[27:38]
	keff_error_float = float(keff_error.replace('E', 'e'))
	keff_error_list.append(keff_error_float)
	file_index_list.append(file_index)

