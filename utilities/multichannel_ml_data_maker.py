import os
import pandas as pd
import tqdm
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

processes = int(input("Enter n. processes: "))
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import ENDF6


First_MT = int(input("Enter first MT number: "))
Second_MT = int(input("Enter second MT number: "))

outputs_directory = input("Enter SCONE output directory: ")
pendf_dir = input("Enter PENDF directory: ")
group_1 = input("Enter group 1: ")
group_2 = input("Enter group 2: ")

parquet_directory = input("Enter parquet directory: ")

output_files = os.listdir(outputs_directory)

keff_list = []
keff_error_list = []

perturbation_1_list = []
perturbation_2_list = []

print("Reading output files...")

for outputfile in tqdm.tqdm(output_files, total=len(output_files)):
	read_object = open(f'{outputs_directory}/{outputfile}', 'r')

	name_split = outputfile.split('_')

	perturbation_1 = float(name_split[1])
	perturbation_2 = float(name_split[-1][:-2])

	perturbation_1_list.append(perturbation_1)
	perturbation_2_list.append(perturbation_2)

	lines = read_object.readlines()
	keffline = lines[12]

	keff_value_string = keffline[15:26]
	keff_value_float = float(keff_value_string.replace('E', 'e'))
	keff_list.append(keff_value_float)

	keff_error = keffline[27:38]
	keff_error_float = float(keff_error.replace('E', 'e'))
	keff_error_list.append(keff_error_float)

keff_dataframe = pd.DataFrame({'keff': keff_list, 'keff_err': keff_error_list, 'p1': perturbation_1_list,
							   'p2': perturbation_2_list})

pendf_names = os.listdir(pendf_dir)
length_list = []

# def parquet_maker(filename):
# 	"""Filename should be the name of the PENDF we're reading from"""
# 	f = open(f'{pendf_dir}/{filename}')
# 	lines = f.readlines()
# 	FirstMTsection = ENDF6.find_section(lines, MF=3, MT=First_MT)
# 	erg, firstxs = ENDF6.read_table(FirstMTsection)
#
#
# 	SecondMTsection = ENDF6.find_section(lines, MF=3, MT=Second_MT)
# 	seconderg, secondxs = ENDF6.read_table(SecondMTsection)
#
# 	name_split = filename.split('_')
# 	coefficient1 = float(name_split[3])
#
# 	coefficient2 = float(name_split[-1][:-6])
# 	coeff1_list = [coefficient1 for i in firstxs]
# 	coeff2_list = [coefficient2 for i in firstxs]
#
# 	reduced_keff_df = keff_dataframe[keff_dataframe.p1 == coefficient1]
# 	reduced_keff_df = reduced_keff_df[reduced_keff_df.p2 == coefficient2]
#
#
# 	keff_list = [reduced_keff_df['keff'].values[0] for i in firstxs]
# 	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in firstxs]
#
# 	df = pd.DataFrame({'ERG': erg,
# 					   'MT18_XS': firstxs,
# 					   'MT2_XS': secondxs,
# 					   'keff': keff_list,
# 					   'keff_err': keff_err_list,
# 					   'p1': coeff1_list,
# 					   'p2': coeff2_list})
#
# 	df.to_parquet(f'Pu-239_g4_MT{FirstMT}_{coefficient1}_MT{SecondMT}_{coefficient2}.parquet', engine='pyarrow')
