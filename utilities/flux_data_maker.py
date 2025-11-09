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


mode = int(input("Enter mode (1/2 perturbations): "))
isotope = input("Enter Element-nucleon_number: ")
MT = int(input("Enter MT number: "))
outputs_directory = input("Enter SCONE output file directory: ")
if mode == 1:
	group = input("Enter group: ")
elif mode == 2: # for 2 groups perturbed simultaneously in the same channel
	group_1 = int(input("Enter group 1: "))
	group_2 = int(input("Enter group 2: "))
destination_directory = os.getcwd()




output_files = os.listdir(outputs_directory)



def extract_flux(output_file):

	if mode == 1:
		if len(output_file) == 14:
			coefficient = float(output_file[7:12])
		elif len(output_file) == 15:
			coefficient = float(output_file[7:13])
	elif mode == 2:
		split_name = output_file.split('_')
		coefficient_g1 = float(split_name[2])
		coefficient_g2 = float(split_name[-1].split('.m')[0])

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

		# plt.figure()
		# plt.plot(active_flux_Res[1][0])
		# plt.plot(active_flux_Res[0][0])
		# plt.grid()
		# plt.ylabel('Flux')
		# plt.title(f'Flux for {coefficient * 100}% perturbation')
		# plt.savefig(f'{saveplot_directory}/{coefficient}.png', dpi=300)
		# plt.show()


		df = pd.DataFrame({'flux': active_flux_Res[0][0], 'flux_std': active_flux_Res[1][0]})

	if mode == 1:
		df_name = f'{isotope}_flux_g{group}_{coefficient:0.3f}_MT{MT}.parquet'
	elif mode == 2:
		df_name = f'{isotope}_flux_g{group_1}_{coefficient_g1:0.3f}_g{group_2}_{coefficient_g2:0.3f}_MT{MT}.parquet'
	df.to_parquet(f'{destination_directory}/{df_name}', engine='pyarrow')


with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(extract_flux, out_file) for out_file in output_files]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass