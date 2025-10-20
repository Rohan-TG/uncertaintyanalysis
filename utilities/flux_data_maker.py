import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

processes = int(input("Enter n. processes: "))
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import ENDF6



# isotope = input("Enter Element-nucleon_number: ")
# MT = int(input("Enter MT number: "))
outputs_directory = input("Enter SCONE output directory: ")
# pendf_dir = input("Enter ERG/XS directory: ")
# group = input("Enter group: ")
# parquet_directory = input("Enter destination directory: ")
#
#
output_files = os.listdir(outputs_directory)

saveplot_directory = input("Enter flux save plot directory: ")


def extract_flux(output_file):

	if len(output_file) == 14:
		coefficient = float(output_file[7:12])
		# perturbation_list.append(coefficient)
	elif len(output_file) == 15:
		coefficient = float(output_file[7:13])
		# perturbation_list.append(coefficient)

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

		plt.figure()
		# plt.plot(active_flux_Res[1][0])
		plt.plot(active_flux_Res[0][0])
		plt.grid()
		plt.ylabel('Flux')
		plt.title(f'Flux for {coefficient * 100}% perturbation')
		plt.savefig(f'{saveplot_directory}/{coefficient}.png', dpi=300)
		# plt.show()

with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(extract_flux, out_file) for out_file in output_files]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass