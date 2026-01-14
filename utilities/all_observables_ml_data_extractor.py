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

processes = int(input("Num. processes: "))
outputs_directory = input("Enter SCONE outputs directory: ")
output_files = os.listdir(outputs_directory)

lower_energy_bound = float(input("Lower energy bound: "))
n_points = int(input("Num. interpolation points below LEB (x for skip): "))


pu239_pendf_directory = input("Pu-239 PENDF directory: ")
pu240_pendf_directory = input("Pu-240 PENDF directory: ")
pu241_pendf_directory = input("Pu-241 PENDF directory: ")

parquet_directory = os.getcwd()



keff_list = []
keff_error_list = []

pu239_file_index_list = []
pu240_file_index_list = []
pu241_file_index_list = []

print("Reading output files...")

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

print('Reading PENDFs and forming dataframes...')

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




	# Interpolation calcs
	if n_points == 'x':
		truncated_fission_erg_239 = []
	else:
		truncated_fission_erg_239 = np.logspace(np.log10(fission_erg_239[0]), np.log10(lower_energy_bound), n_points) # This is x_coarse
		truncated_fission_erg_239 = truncated_fission_erg_239.tolist()


	truncated_fission_xs_239 = []
	for ERG, xsval in zip(fission_erg_239, fission_xs_239):
		if ERG >= lower_energy_bound:
			truncated_fission_erg_239.append(ERG)
			truncated_fission_xs_239.append(xsval)

	capture_to_fission_239 = np.interp(truncated_fission_erg_239, capture_erg_239, capture_xs_239)
	n2n_to_fission_239 = np.interp(truncated_fission_erg_239, n2n_erg_239, n2n_xs_239)
	inelastic_to_fission_239 = np.interp(truncated_fission_erg_239, inelastic_erg_239, inelastic_xs_239)
	elastic_to_fission_239 = np.interp(truncated_fission_erg_239, elastic_erg_239, elastic_xs_239)




	capture_to_fission_240 = np.interp(truncated_fission_erg_239, capture_erg_240, capture_xs_240)
	n2n_to_fission_240 = np.interp(truncated_fission_erg_239, n2n_erg_240, n2n_xs_240)
	inelastic_to_fission_240 = np.interp(truncated_fission_erg_239, inelastic_erg_240, inelastic_xs_240)
	elastic_to_fission_240 = np.interp(truncated_fission_erg_239, elastic_erg_240, elastic_xs_240)
	fission_to_fission_240 = np.interp(truncated_fission_erg_239, fission_erg_240, fission_xs_240)


	capture_to_fission_241 = np.interp(truncated_fission_erg_239, capture_erg_241, capture_xs_241)
	n2n_to_fission_241 = np.interp(truncated_fission_erg_239, n2n_erg_241, n2n_xs_241)
	inelastic_to_fission_241 = np.interp(truncated_fission_erg_239, inelastic_erg_241, inelastic_xs_241)
	elastic_to_fission_241 = np.interp(truncated_fission_erg_239, elastic_erg_241, elastic_xs_241)
	fission_to_fission_241 = np.interp(truncated_fission_erg_239, fission_erg_241, fission_xs_241)

	# extract keff data
	keff_list = [reduced_keff_df['keff'].values[0] for i in truncated_fission_erg_239]
	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in truncated_fission_erg_239]



	df = pd.DataFrame({'ERG': truncated_fission_erg_239,
					   '94239_MT2_XS': elastic_to_fission_239,
					   '94239_MT4_XS': inelastic_to_fission_239,
					   '94239_MT16_XS': n2n_to_fission_239,
					   '94239_MT18_XS': truncated_fission_xs_239,
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

	df.to_parquet(f'{parquet_directory}/Random_all_channels_Pu-239_{pu239_index}_Pu-240_{pu240_index}_Pu-241_{pu241_index}.parquet',
				  engine='pyarrow')



# run the whole thing
with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(parquet_maker, indices) for indices in index_matrix
	]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass