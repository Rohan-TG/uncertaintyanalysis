import pandas as pd
import tqdm
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


processes = int(input("Enter n. processes: "))
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Reactions
import ENDF6

# k_eff directory
isotope = input("Enter Element-nucleon_number: ")
MT = int(input("Enter MT number: "))
outputs_directory = input("Enter SCONE output directory: ")
pendf_dir = input("Enter PENDF directory: ")
group1 = input("Enter group 1: ")
group2 = input("Enter group 2: ")
parquet_directory = os.getcwd()



output_files = os.listdir(outputs_directory)

keff_list = []
keff_error_list = []
perturbation_g1_list = []
perturbation_g2_list = []

print("Reading output files...")
for outputfile in tqdm.tqdm(output_files, total=len(output_files)):
	read_object = open(f'{outputs_directory}/{outputfile}', 'r')

	split_name = outputfile.split('_')

	coefficient1 = float(split_name[2])
	perturbation_g1_list.append(coefficient1)

	coefficient2 = float(split_name[-1].split('.m')[0])
	perturbation_g2_list.append(coefficient2)

	lines = read_object.readlines()
	keffline = lines[12]

	keff_value_string = keffline[15:26]
	keff_value_float = float(keff_value_string.replace('E', 'e'))
	keff_list.append(keff_value_float)

	keff_error = keffline[27:38]
	keff_error_float = float(keff_error.replace('E', 'e'))
	keff_error_list.append(keff_error_float)


keff_dataframe = pd.DataFrame({'keff': keff_list, 'keff_err': keff_error_list, 'p1': perturbation_g1_list,
							   'p2': perturbation_g2_list,
							   })


# PENDF directory


pendf_names = os.listdir(pendf_dir)
length_list = []

print('Reading PENDFs and forming dataframes...')

def parquet_maker(filename):
	"""Filename should be the name of the PENDF we're reading from"""
	f = open(f'{pendf_dir}/{filename}')
	lines = f.readlines()
	FirstMTsection = ENDF6.find_section(lines, MF=3, MT=MT)
	erg, firstxs = ENDF6.read_table(FirstMTsection)

	name_split = filename.split('_')
	coefficient_1 = float(name_split[2])

	coefficient_2 = float(name_split[4])

	coeff1_list = [coefficient_1 for i in firstxs]
	coeff2_list = [coefficient_2 for i in firstxs]

	reduced_keff_df = keff_dataframe[(keff_dataframe.p1 == coefficient_1) & (keff_dataframe.p2 == coefficient_2)]

	keff_list = [reduced_keff_df['keff'].values[0] for i in firstxs]
	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in firstxs]

	df = pd.DataFrame({'ERG': erg,
					   'XS': firstxs,
					   'keff': keff_list,
					   'keff_err': keff_err_list,
					   'p1': coeff1_list,
					   'p2': coeff2_list,
					   })

	df.to_parquet(f'{parquet_directory}/{isotope}_g{group1}_{coefficient1:0.3f}_g{group2}_{coefficient2:0.3f}_MT{MT}.parquet',
				  engine='pyarrow')




with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(parquet_maker, file) for file in pendf_names]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass









# for filename in tqdm.tqdm(pendf_names, total=len(pendf_names)):
# 	f = open(f'{pendf_dir}/{filename}')
# 	lines = f.readlines()
# 	FirstMTsection = ENDF6.find_section(lines, MF=3, MT=MT)
# 	erg, firstxs = ENDF6.read_table(FirstMTsection)
#
#
# 	# SecondMTsection = ENDF6.find_section(lines, MF=3, MT=SecondMT)
# 	# seconderg, secondxs = ENDF6.read_table(SecondMTsection)
#
# 	name_split = filename.split('_')
# 	coefficient = float(name_split[2])
#
# 	# coefficient2 = float(name_split[-1][:-6])
# 	coeff_list = [coefficient for i in firstxs]
# 	# coeff2_list = [coefficient2 for i in firstxs]
#
# 	reduced_keff_df = keff_dataframe[keff_dataframe.p == coefficient]
# 	# reduced_keff_df = reduced_keff_df[reduced_keff_df.p2 == coefficient2]
#
#
# 	keff_list = [reduced_keff_df['keff'].values[0] for i in firstxs]
# 	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in firstxs]
#
# 	df = pd.DataFrame({'ERG': erg,
# 					   'XS': firstxs,
# 					   # 'MT2_XS': secondxs,
# 					   'keff': keff_list,
# 					   'keff_err': keff_err_list,
# 					   'p': coeff_list,
# 					   # 'p2': coeff2_list,
# 					   })
#
# 	df.to_parquet(f'{parquet_directory}/Pu-239_g{group}_{coefficient:0.3f}_MT{MT}.parquet', engine='pyarrow')

	# df.to_csv(f'csvs/g1_Pu9_{coefficient:0.3f}_MT18.csv')
	# df_temp = pd.DataFrame({'ERG': erg, 'XS': xs, 'P':coeff_list})


	# df_temp.to_csv(f'Pu239_{coefficient}_flat_MT18_XS.csv')


