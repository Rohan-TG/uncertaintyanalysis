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
from concurrent.futures import ProcessPoolExecutor, as_completed

isotope = input("Enter Element-nucleon_number: ")
MT = int(input("Enter MT number: "))
outputs_directory = input("Enter SCONE output directory: ")
pendf_dir = input("Enter PENDF directory: ")
parquet_directory = os.getcwd()


output_files = os.listdir(outputs_directory)

keff_list = []
keff_error_list = []
file_index_list = []


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
	file_index_list.append(file_index)

keff_dataframe = pd.DataFrame({'keff': keff_list,
							   'keff_err': keff_error_list,
							   'file_index': file_index_list})

pendf_names = os.listdir(pendf_dir)
length_list = []

print('Reading PENDFs and forming dataframes...')

def parquet_maker(filename):
	"""Filename should be the name of the PENDF we're reading from"""
	f = open(f'{pendf_dir}/{filename}')
	lines = f.readlines()
	FirstMTsection = ENDF6.find_section(lines, MF=3, MT=MT)
	erg, firstxs = ENDF6.read_table(FirstMTsection)

	pendf_index = int(filename.split('.pendf')[0].split('_')[1])



	reduced_keff_df = keff_dataframe[keff_dataframe.file_index == pendf_index]

	keff_list = [reduced_keff_df['keff'].values[0] for i in firstxs]
	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in firstxs]

	df = pd.DataFrame({'ERG': erg,
					   'XS': firstxs,
					   'keff': keff_list,
					   'keff_err': keff_err_list,
					   })

	df.to_parquet(f'{parquet_directory}/{isotope}_random_{pendf_index}_MT{MT}.parquet', engine='pyarrow')


with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(parquet_maker, file) for file in pendf_names]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass
