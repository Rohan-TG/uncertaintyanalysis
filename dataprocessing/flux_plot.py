import matplotlib.pyplot as plt
import numpy as np



def extract_flux(output_file):
	"""output_file: aboslute location of file"""

	lines = output_file.readlines()

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



with open('output-normal.m', 'r') as file:
	lines = file.readlines()

	active_flux_EnergyBounds = lines[16]
	active_flux_EnergyBounds = active_flux_EnergyBounds.split('[')
	active_flux_EnergyBounds = active_flux_EnergyBounds[-1].split(']')
	active_flux_EnergyBounds = active_flux_EnergyBounds[0].split(',')
	afE = [float(i) for i in active_flux_EnergyBounds]
	energy_bounds = np.array(afE, dtype=float).reshape((2,1,300), order='F')


	active_flux_Res = lines[17]
	active_flux_Res = active_flux_Res.split('[')
	active_flux_Res = active_flux_Res[-1].split(']')
	active_flux_Res = active_flux_Res[0].split(',')
	afR = [float(i) for i in active_flux_Res]

	active_flux_Res = np.array(afR, dtype=float).reshape((2, 1, 300), order='F')

	plt.figure()
	plt.plot(active_flux_Res[1][0])
	plt.grid()
	plt.ylabel('Flux')
	plt.show()










