import pandas as pd
import tqdm
import os
import sys
sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
from groupEnergies import Reactions
import ENDF6


outputs_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/scone_benchmarks/multi-perturbation/fission_elastic/pu9/g4g4/outputfiles'

FirstMT = Reactions.fission
SecondMT = Reactions.elastic

output_files = os.listdir(outputs_directory)

keff_list = []
keff_error_list = []
perturbation_1_list = []
perturbation_2_list = []

for outputfile in tqdm.tqdm(output_files, total=len(output_files)):
	read_object = open(f'{outputs_directory}/{outputfile}', 'r')

	name_split = outputfile.split('_')

	perturbation_1 = float(name_split[2])
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



pendf_dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/dataprocessing/multichannel/fission_elastic/Pu239/g4/pendf'

pendf_names = os.listdir(pendf_dir)
length_list = []


for filename in tqdm.tqdm(pendf_names, total=len(pendf_names)):
	f = open(f'{pendf_dir}/{filename}')
	lines = f.readlines()
	FirstMTsection = ENDF6.find_section(lines, MF=3, MT=FirstMT)
	erg, firstxs = ENDF6.read_table(FirstMTsection)


	SecondMTsection = ENDF6.find_section(lines, MF=3, MT=SecondMT)
	seconderg, secondxs = ENDF6.read_table(SecondMTsection)

	name_split = filename.split('_')
	coefficient1 = float(name_split[3])

	coefficient2 = float(name_split[-1][:-6])
	coeff1_list = [coefficient1 for i in firstxs]
	coeff2_list = [coefficient2 for i in firstxs]

	reduced_keff_df = keff_dataframe[keff_dataframe.p1 == coefficient1]
	reduced_keff_df = reduced_keff_df[reduced_keff_df.p2 == coefficient2]


	keff_list = [reduced_keff_df['keff'].values[0] for i in firstxs]
	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in firstxs]

	df = pd.DataFrame({'ERG': erg,
					   'MT18_XS': firstxs,
					   'MT2_XS': secondxs,
					   'keff': keff_list,
					   'keff_err': keff_err_list,
					   'p1': coeff1_list,
					   'p2': coeff2_list})

	break

	# df.to_csv(f'csvs/g1_Pu9_{coefficient:0.3f}_MT18.csv')
	# df_temp = pd.DataFrame({'ERG': erg, 'XS': xs, 'P':coeff_list})


	# df_temp.to_csv(f'Pu239_{coefficient}_flat_MT18_XS.csv')
