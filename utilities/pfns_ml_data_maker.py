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
from concurrent.futures import ProcessPoolExecutor, as_completed
import endf

processes = int(input("Num. processes: "))
outputs_directory = input("Enter SCONE outputs directory: ")
output_files = os.listdir(outputs_directory)

destination_directory = os.getcwd()

pu9_endf6_directory = input("Pu-239 ENDF6 directory: ")
pu0_endf6_directory = input("Pu-240 ENDF6 directory: ")
pu1_endf6_directory = input("Pu-241 ENDF6 directory: ")

# pu9_target_dir = ''

keff_list = []
keff_error_list = []

pu239_file_index_list = []
pu240_file_index_list = []
pu241_file_index_list = []

print("\nReading output files...")

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

	pu239_file_index_list.append(Pu239_file_index)
	pu240_file_index_list.append(Pu240_file_index)
	pu241_file_index_list.append(Pu241_file_index)


index_matrix = []
for pu9_idx, pu0_idx, pu1_idx in zip(pu239_file_index_list, pu240_file_index_list, pu241_file_index_list):
	index_matrix.append([pu9_idx, pu0_idx, pu1_idx])


keff_dataframe = pd.DataFrame({'keff': keff_list,
							   'keff_err': keff_error_list,
							   'pu239_file_index': pu239_file_index_list,
							   'pu240_file_index': pu240_file_index_list,
							   'pu241_file_index': pu241_file_index_list,})



print('\nReading PENDFs and forming dataframes...')


def scrape_pfns(endf6_file, MT=18):
	mat = endf.Material(endf6_file)

	incident_energies = mat.section_data[5, MT]['subsections'][0]['distribution']['E']

	y_matrix = []
	for i, set in enumerate(mat.section_data[5, MT]['subsections'][0]['distribution']['g']):
		y_values = mat.section_data[5, MT]['subsections'][0]['distribution']['g'][i].y
		y_matrix.append(y_values)

		x = mat.section_data[5, MT]['subsections'][0]['distribution']['g'][i].x

	return incident_energies, y_matrix, x




def parquet_maker(outputfile):

	pu9_index = int(outputfile.split('.m')[0].split('_')[1])
	pu0_index = int(outputfile.split('.m')[0].split('_')[3])
	pu1_index = int(outputfile.split('.m')[0].split('_')[5])

	pu9_endf6_file = f'{pu9_endf6_directory}/94239_{pu9_index}.endf6'
	pu0_endf6_file = f'{pu0_endf6_directory}/94240_{pu0_index}.endf6'
	pu1_endf6_file = f'{pu1_endf6_directory}/94241_{pu1_index}.endf6'

	incident_energies_pu9, y_values_pu9, pfns_grid_pu9 = scrape_pfns(pu9_endf6_file)
	incident_energies_pu0, y_values_pu0, pfns_grid_pu0 = scrape_pfns(pu0_endf6_file)
	incident_energies_pu1, y_values_pu1, pfns_grid_pu1 = scrape_pfns(pu1_endf6_file)

	pu9_savefilename = f'94239_{pu9_index}_MF5_data.parquet'
	pu0_savefilename = f'94240_{pu0_index}_MF5_data.parquet'
	pu1_savefilename = f'94241_{pu1_index}_MF5_data.parquet'


	pu9_df = pd.DataFrame(np.asarray(y_values_pu9).transpose(), columns = incident_energies_pu9)
	pu0_df = pd.DataFrame(np.asarray(y_values_pu0).transpose(), columns = incident_energies_pu0)
	pu1_df = pd.DataFrame(np.asarray(y_values_pu1).transpose(), columns = incident_energies_pu1)

	pu9_df.to_parquet(f'{destination_directory}/{pu9_savefilename}', engine='pyarrow')
	pu0_df.to_parquet(f'{destination_directory}/{pu0_savefilename}', engine='pyarrow')
	pu1_df.to_parquet(f'{destination_directory}/ {pu1_savefilename}', engine='pyarrow')


# run the whole thing
print('\nRunning')

with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(parquet_maker, outputfile) for outputfile in output_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass