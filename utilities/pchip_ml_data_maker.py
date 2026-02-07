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
from scipy.interpolate import PchipInterpolator

processes = int(input("Num. processes: "))
outputs_directory = input("Enter SCONE outputs directory: ")
output_files = os.listdir(outputs_directory)

interpolation_energy_bound = float(input("Interpolation bound: "))
relative_tolerance = float(input("Reconstruction tolerance: "))


pu239_pendf_directory = input("Pu-239 PENDF directory: ")
pu240_pendf_directory = input("Pu-240 PENDF directory: ")
pu241_pendf_directory = input("Pu-241 PENDF directory: ")

parquet_directory = os.getcwd()



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


pu239_pendf_names = os.listdir(pu239_pendf_directory)
pu240_pendf_names = os.listdir(pu240_pendf_directory)
pu241_pendf_names = os.listdir(pu241_pendf_directory)


print('\nDefining PCHIP function...')

def thin_relative_error_logx(x, y, rel_tol=0.999, y_floor=None, max_points=None):
	x = np.asarray(x)
	y = np.asarray(y)
	idx = np.argsort(x)
	x = x[idx]
	y = y[idx]

	# log-x coordinate
	u = np.log(x)

	if y_floor is None:
		y_floor = 1e-12 * np.nanmax(y)

	# Keep array
	keep = np.zeros_like(y, dtype=bool)
	keep[0] = True
	keep[-1] = True

	# Stack of segments as (i, j)
	stack = [(0, len(y)-1)]

	while stack:
		i, j = stack.pop()
		if j <= i + 1:
			continue

		# Build interpolant from endpoints only
		interp = PchipInterpolator(u[[i, j]], y[[i, j]], extrapolate=False)

		k = np.arange(i+1, j)
		yhat = interp(u[k])

		denom = np.maximum(np.abs(y[k]), y_floor)
		err = np.abs(yhat - y[k]) / denom

		m = np.argmax(err)
		if err[m] > rel_tol:
			km = k[m]
			keep[km] = True

			# Split and continue
			stack.append((i, km))
			stack.append((km, j))

			# stop if points budget expended
			if max_points is not None and keep.sum() >= max_points:
				break

	# Return subset in original order
	kept_idx_sorted = np.nonzero(keep)[0]
	kept_idx = idx[kept_idx_sorted]  # indices into original arrays
	return kept_idx, x[kept_idx_sorted], y[kept_idx_sorted]


def pchip_initialiser():

	f239 = open(f'{pu239_pendf_directory}/{pu239_pendf_names[0]}', 'r')
	lines_239 = f239.readlines()
	fission_section_239 = ENDF6.find_section(lines_239, MF=3, MT=Reactions.fission)
	fission_erg_239, fission_xs_239 = ENDF6.read_table(fission_section_239)

	truncated_erg_mt18_239 = []
	truncated_xs_mt18_239 = []

	# fast_energy_mt18_239 = [erg for erg in fission_erg_239 if erg >= interpolation_energy_bound]
	# fast_xs_mt18_239 = [xs for xs, erg in zip(fission_xs_239, fission_erg_239) if erg >= interpolation_energy_bound]

	for erg, xs in zip(fission_erg_239, fission_xs_239):
		if erg <= interpolation_energy_bound:
			truncated_erg_mt18_239.append(erg)
			truncated_xs_mt18_239.append(xs)

	kept_idx, thinned_erg, thinned_xs = thin_relative_error_logx(x=truncated_erg_mt18_239,
																 y=truncated_xs_mt18_239,
																 rel_tol=relative_tolerance)
	return(thinned_erg) #

interpolation_energies = pchip_initialiser()


print('\nReading PENDFs and forming dataframes...')

def parquet_maker(index_combination):
	"""Filename should be the name of the PENDF we're reading from"""


	pu239_index = index_combination[0]
	pu240_index = index_combination[1]
	pu241_index = index_combination[2]

	reduced_keff_df = keff_dataframe[keff_dataframe.pu239_file_index == pu239_index]
	# reduced_keff_df = reduced_keff_df[reduced_keff_df.pu240_file_index == pu240_index]
	# reduced_keff_df = reduced_keff_df[reduced_keff_df.pu241_file_index == pu241_index]

	pu239_filename = f'94239_{pu239_index}.pendf'
	pu240_filename = f'94240_{pu240_index}.pendf'
	pu241_filename = f'94241_{pu241_index}.pendf'


	# Extract Pu-239 data
	f239 = open(f'{pu239_pendf_directory}/{pu239_filename}')
	lines_239 = f239.readlines()

	elastic_section_239 = ENDF6.find_section(lines_239, MF=3, MT=Reactions.elastic)
	elastic_erg_239, elastic_xs_239 = ENDF6.read_table(elastic_section_239)

	inelastic_section_239 = ENDF6.find_section(lines_239, MF=3, MT=Reactions.inelastic)
	inelastic_erg_239, inelastic_xs_239 = ENDF6.read_table(inelastic_section_239)

	n2n_section_239 = ENDF6.find_section(lines_239, MF=3, MT=Reactions.n2n)
	n2n_erg_239, n2n_xs_239 = ENDF6.read_table(n2n_section_239)

	fission_section_239 = ENDF6.find_section(lines_239, MF=3, MT=Reactions.fission)
	fission_erg_239, fission_xs_239 = ENDF6.read_table(fission_section_239)

	capture_section_239 = ENDF6.find_section(lines_239, MF=3, MT=Reactions.capture)
	capture_erg_239, capture_xs_239 = ENDF6.read_table(capture_section_239)


	# Form energy grid using PCHIP function

	fast_energy_mt18_239 = [erg for erg in fission_erg_239 if erg >= interpolation_energy_bound]
	fast_xs_mt18_239 = [xs for xs, erg in zip(fission_xs_239, fission_erg_239) if erg >= interpolation_energy_bound]



	# Extract Pu-240 data
	f240 = open(f'{pu240_pendf_directory}/{pu240_filename}')
	lines_240 = f240.readlines()

	elastic_section_240 = ENDF6.find_section(lines_240, MF=3, MT=Reactions.elastic)
	elastic_erg_240, elastic_xs_240 = ENDF6.read_table(elastic_section_240)

	inelastic_section_240 = ENDF6.find_section(lines_240, MF=3, MT=Reactions.inelastic)
	inelastic_erg_240, inelastic_xs_240 = ENDF6.read_table(inelastic_section_240)

	n2n_section_240 = ENDF6.find_section(lines_240, MF=3, MT=Reactions.n2n)
	n2n_erg_240, n2n_xs_240 = ENDF6.read_table(n2n_section_240)

	fission_section_240 = ENDF6.find_section(lines_240, MF=3, MT=Reactions.fission)
	fission_erg_240, fission_xs_240 = ENDF6.read_table(fission_section_240)

	capture_section_240 = ENDF6.find_section(lines_240, MF=3, MT=Reactions.capture)
	capture_erg_240, capture_xs_240 = ENDF6.read_table(capture_section_240)




	# Extract Pu-241 data
	f241 = open(f'{pu241_pendf_directory}/{pu241_filename}')
	lines_241 = f241.readlines()

	elastic_section_241 = ENDF6.find_section(lines_241, MF=3, MT=Reactions.elastic)
	elastic_erg_241, elastic_xs_241 = ENDF6.read_table(elastic_section_241)

	inelastic_section_241 = ENDF6.find_section(lines_241, MF=3, MT=Reactions.inelastic)
	inelastic_erg_241, inelastic_xs_241 = ENDF6.read_table(inelastic_section_241)

	n2n_section_241 = ENDF6.find_section(lines_241, MF=3, MT=Reactions.n2n)
	n2n_erg_241, n2n_xs_241 = ENDF6.read_table(n2n_section_241)

	fission_section_241 = ENDF6.find_section(lines_241, MF=3, MT=Reactions.fission)
	fission_erg_241, fission_xs_241 = ENDF6.read_table(fission_section_241)

	capture_section_241 = ENDF6.find_section(lines_241, MF=3, MT=Reactions.capture)
	capture_erg_241, capture_xs_241 = ENDF6.read_table(capture_section_241)




	full_thinned_erg = interpolation_energies + fast_energy_mt18_239 # full energy grid after applying PCHIP

	full_thinned_xs = np.interp(full_thinned_erg, fission_erg_239, fission_xs_239)

	# full_thinned_xs = thinned_xs + fast_energy_mt18_239

	# truncated_fission_xs_239 = np.interp(full_thinned_erg, fission_erg_239, fission_xs_239)


	capture_to_fission_239 = np.interp(full_thinned_erg, capture_erg_239, capture_xs_239)
	n2n_to_fission_239 = np.interp(full_thinned_erg, n2n_erg_239, n2n_xs_239)
	inelastic_to_fission_239 = np.interp(full_thinned_erg, inelastic_erg_239, inelastic_xs_239)
	elastic_to_fission_239 = np.interp(full_thinned_erg, elastic_erg_239, elastic_xs_239)




	capture_to_fission_240 = np.interp(full_thinned_erg, capture_erg_240, capture_xs_240)
	n2n_to_fission_240 = np.interp(full_thinned_erg, n2n_erg_240, n2n_xs_240)
	inelastic_to_fission_240 = np.interp(full_thinned_erg, inelastic_erg_240, inelastic_xs_240)
	elastic_to_fission_240 = np.interp(full_thinned_erg, elastic_erg_240, elastic_xs_240)
	fission_to_fission_240 = np.interp(full_thinned_erg, fission_erg_240, fission_xs_240)


	capture_to_fission_241 = np.interp(full_thinned_erg, capture_erg_241, capture_xs_241)
	n2n_to_fission_241 = np.interp(full_thinned_erg, n2n_erg_241, n2n_xs_241)
	inelastic_to_fission_241 = np.interp(full_thinned_erg, inelastic_erg_241, inelastic_xs_241)
	elastic_to_fission_241 = np.interp(full_thinned_erg, elastic_erg_241, elastic_xs_241)
	fission_to_fission_241 = np.interp(full_thinned_erg, fission_erg_241, fission_xs_241)

	# extract keff data
	keff_list = [reduced_keff_df['keff'].values[0] for i in full_thinned_erg]
	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in full_thinned_erg]



	df = pd.DataFrame({'ERG': full_thinned_erg,
					   '94239_MT2_XS': elastic_to_fission_239,
					   '94239_MT4_XS': inelastic_to_fission_239,
					   '94239_MT16_XS': n2n_to_fission_239,
					   '94239_MT18_XS': full_thinned_xs, # bug
					   '94239_MT102_XS': capture_to_fission_239,
					   '94240_MT2_XS': elastic_to_fission_240,
					   '94240_MT4_XS': inelastic_to_fission_240,
					   '94240_MT16_XS': n2n_to_fission_240,
					   '94240_MT18_XS': fission_to_fission_240,
					   '94240_MT102_XS': capture_to_fission_240,
					   '94241_MT2_XS': elastic_to_fission_241,
					   '94241_MT4_XS': inelastic_to_fission_241,
					   '94241_MT16_XS': n2n_to_fission_241,
					   '94241_MT18_XS': fission_to_fission_241,
					   '94241_MT102_XS': capture_to_fission_241,
					   'keff': keff_list,
					   'keff_err': keff_err_list,
					   })

	df.to_parquet(f'{parquet_directory}/Thinned_energy_random_Pu-239_{pu239_index}_Pu-240_{pu240_index}_Pu-241_{pu241_index}.parquet',
				  engine='pyarrow')



# run the whole thing
with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(parquet_maker, indices) for indices in index_matrix
	]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass