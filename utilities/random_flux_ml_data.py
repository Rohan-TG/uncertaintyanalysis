import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import os
import sys
import numpy as np

processes = int(input("Enter n. processes: "))
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')

isotope = input("Enter Element-nucleon_number: ")
MT = int(input("Enter MT number: "))
outputs_directory = input("Enter SCONE output file directory: ")

destination_directory = os.getcwd()


output_files = os.listdir(outputs_directory)



def extract_flux(output_file):
	file_index = int(output_file.split('.m')[0].split('-')[1])

	with open(f'{outputs_directory}/{output_file}', 'r') as f:
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

		df = pd.DataFrame({'flux': active_flux_Res[0][0], 'flux_std': active_flux_Res[1][0]})

	df_name = f'{isotope}_flux_random_{int(file_index)}_MT{MT}.parquet'
	df.to_parquet(f'{destination_directory}/{df_name}', engine='pyarrow')


with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(extract_flux, out_file) for out_file in output_files]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass