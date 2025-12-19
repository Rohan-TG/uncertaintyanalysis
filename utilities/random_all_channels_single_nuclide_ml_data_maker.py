import os
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Reactions
import pandas as pd
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import ENDF6
import numpy as np

isotope = input("Enter Element-nucleon_number: ")
outputs_directory = input("Enter SCONE outputs directory: ")
output_files = os.listdir(outputs_directory)

lower_energy_bound = float(input("Truncation: "))

pendf_directory = input("Enter PENDF directory: ")
processes = int(input("Number of processes: "))

parquet_directory = os.getcwd()

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

pendf_names = os.listdir(pendf_directory)
length_list = []
print('Reading PENDFs and forming dataframes...')

def parquet_maker(filename):
	"""Filename should be the name of the PENDF we're reading from"""
	f = open(f'{pendf_directory}/{filename}')
	lines = f.readlines()

	# Channels 2 elastic / 4 inelastic / 16 n,2n / 17 n,3n / 18 fission / 102 capture /

	elastic_section = ENDF6.find_section(lines, MF=3, MT=Reactions.elastic)
	elastic_erg, elastic_xs = ENDF6.read_table(elastic_section)


	inelastic_section = ENDF6.find_section(lines, MF=3, MT=Reactions.inelastic)
	inelastic_erg, inelastic_xs = ENDF6.read_table(inelastic_section)


	n2n_section = ENDF6.find_section(lines, MF=3, MT=Reactions.n2n)
	n2n_erg, n2n_xs = ENDF6.read_table(n2n_section)


	fission_section = ENDF6.find_section(lines, MF=3, MT=Reactions.fission)
	fission_erg, fission_xs = ENDF6.read_table(fission_section)


	capture_section = ENDF6.find_section(lines, MF=3, MT=Reactions.capture)
	capture_erg, capture_xs = ENDF6.read_table(capture_section)


	pendf_index = int(filename.split('.pendf')[0].split('_')[1])
	reduced_keff_df = keff_dataframe[keff_dataframe.file_index == pendf_index]


	truncated_fission_erg = []
	truncated_fission_xs = []
	for ERG, xsval in zip(fission_erg, fission_xs):
		if ERG >= lower_energy_bound:
			truncated_fission_erg.append(ERG)
			truncated_fission_xs.append(xsval)

	capture_to_fission = np.interp(truncated_fission_erg, capture_erg, capture_xs)
	n2n_to_fission = np.interp(truncated_fission_erg, n2n_erg, n2n_xs)
	inelastic_to_fission = np.interp(truncated_fission_erg, inelastic_erg, inelastic_xs)
	elastic_to_fission = np.interp(truncated_fission_erg, elastic_erg, elastic_xs)






	keff_list = [reduced_keff_df['keff'].values[0] for i in truncated_fission_erg]
	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in truncated_fission_erg]

	df = pd.DataFrame({'ERG': fission_erg,
					   'MT2_XS': elastic_to_fission,
					   'MT4_XS': inelastic_to_fission,
					   'MT16_XS': n2n_to_fission,
					   'MT18_XS': truncated_fission_xs,
					   'MT102_XS': capture_to_fission,
					   'keff': keff_list,
					   'keff_err': keff_err_list,
					   })

	df.to_parquet(f'{parquet_directory}/{isotope}_random_{pendf_index}_all_channels.parquet', engine='pyarrow')

with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(parquet_maker, file) for file in pendf_names]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass
