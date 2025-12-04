import os
processes = int(input("Enter n. processes: "))
computer = os.uname().nodename
import sys
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')

import pandas as pd
import ENDF6
import tqdm

isotope = input("Enter Element-nucleon_number: ")
MT = int(input("Enter MT number: "))
outputs_directory = input("Enter SCONE output directory: ")
pendf_dir = input("Enter PENDF directory: ")
parquet_directory = os.getcwd()


output_files = os.listdir(outputs_directory)

keff_list = []
keff_error_list = []

print("Reading output files...")
for outputfile in tqdm.tqdm(output_files, total=len(output_files)):
	read_object = open(f'{outputs_directory}/{outputfile}', 'r')

	file_index = int(outputfile.split('.m')[0].split('-')[1])



	lines = read_object.readlines()
	keffline = lines[12]

	keff_value_string = keffline[15:26]
	keff_value_float = float(keff_value_string.replace('E', 'e'))
	keff_list.append(keff_value_float)

	keff_error = keffline[27:38]
	keff_error_float = float(keff_error.replace('E', 'e'))
	keff_error_list.append(keff_error_float)

