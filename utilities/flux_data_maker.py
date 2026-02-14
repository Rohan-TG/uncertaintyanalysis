import os
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
from tqdm import tqdm
import numpy as np


processes = int(input("Processes: "))
outputs_directory = input("SCONE output directory: ")
output_files = os.listdir(outputs_directory)
destination_directory = input('Destination dir. (. for here): ')
if destination_directory == '.':
	destination_directory = os.getcwd()


for file in tqdm(output_files, total=len(output_files)):
	with open(f'{outputs_directory}/{file}', 'r') as f:
		Pu239_file_index = int(file.split('.m')[0].split('_')[1])

		Pu240_file_index = int(file.split('.m')[0].split('_')[3])

		Pu241_file_index = int(file.split('.m')[0].split('_')[5])

		lines = f.readlines()
		active_flux_EnergyBounds = lines[16]
		active_flux_EnergyBounds = active_flux_EnergyBounds.split('[')
		active_flux_EnergyBounds = active_flux_EnergyBounds[-1].split(']')
		active_flux_EnergyBounds = active_flux_EnergyBounds[0].split(',')
		afE = [float(i) for i in active_flux_EnergyBounds]
		energy_bounds = np.array(afE, dtype=float).reshape((2, 1, 300), order='F')

		active_flux_Res = lines[17]
		active_flux_Res = active_flux_Res.split('[')
		active_flux_Res = active_flux_Res[-1].split(']')
		active_flux_Res = active_flux_Res[0].split(',')
		afR = [float(i) for i in active_flux_Res]

		active_flux_Res = np.array(afR, dtype=float).reshape((2, 1, 300), order='F')

		keffline = lines[12]

		keff_value_string = keffline[15:26]
		keff_value_float = float(keff_value_string.replace('E', 'e'))

		keff_error = keffline[27:38]
		keff_error_float = float(keff_error.replace('E', 'e'))

		raw_flux = active_flux_Res[0][0]
		flux_error = active_flux_Res[1][0]

		df = pd.DataFrame({'flux': raw_flux,
						   'flux_errror': flux_error,
						   'low_erg_bounds': energy_bounds[0][0],
						   'high_erg_bounds': energy_bounds[1][0],})

		filename = f'Flux_data_Pu-239_{Pu239_file_index}_Pu-240_{Pu240_file_index}_Pu-241_{Pu241_file_index}'
		df.to_parquet(f'{destination_directory}/{filename}.parquet')