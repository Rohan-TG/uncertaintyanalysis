import pandas as pd
import tqdm
import sys
sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
import ENDF6
import os

# Scrape k_eff data from scone runs

scone_run_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/scone_benchmarks/flatmt18/Pu0/outputfiles'

output_files = os.listdir(scone_run_directory)

keff_list = []
keff_error_list = []
coeff_list = []

for filename in tqdm.tqdm(output_files, total=len(output_files)):
	obj = open(f'{scone_run_directory}/{filename}')
	if len(filename) == 14:
		coefficient = float(filename[7:12])
		coeff_list.append(coefficient)
	elif len(filename) == 15:
		coefficient = float(filename[7:13])
		coeff_list.append(coefficient)

	lines = obj.readlines()
	keffline = lines[12]

	keff_value_string = keffline[15:26]
	keff_value_float = float(keff_value_string.replace('E', 'e'))
	keff_list.append(keff_value_float)

	keff_error = keffline[27:38]
	keff_error_float = float(keff_error.replace('E', 'e'))
	keff_error_list.append(keff_error_float)

k_df = pd.DataFrame({'keff':keff_list, 'keff_err':keff_error_list, 'p': coeff_list})

print("K_eff values retrieved...")
# ======================================================================================================================



pendf_dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/dataprocessing/Pu240/flat_perturbations/fission/pendf'

pendf_names = os.listdir(pendf_dir)
length_list = []

MT = 18

for filename in tqdm.tqdm(pendf_names, total=len(pendf_names)):
	f = open(f'{pendf_dir}/{filename}')
	lines = f.readlines()
	section = ENDF6.find_section(lines, MF=3, MT=MT)
	erg, xs = ENDF6.read_table(section)

	name_split = filename.split('_')
	coefficient = float(name_split[2])
	coeff_list = [coefficient for i in xs]

	reduced_keff_df = k_df[k_df.p == coefficient]
	keff_list = [reduced_keff_df['keff'].values[0] for i in xs]
	keff_err_list = [reduced_keff_df['keff_err'].values[0] for i in xs]

	df = pd.DataFrame({'ERG': erg,
					   'XS': xs,
					   'keff': keff_list,
					   'keff_err': keff_err_list,
					   'p': coeff_list})


print('Dataframes created.')