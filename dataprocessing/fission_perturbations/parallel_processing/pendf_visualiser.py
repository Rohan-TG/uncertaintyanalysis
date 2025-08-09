import matplotlib.pyplot as plt
import pandas as pd

import ENDF6
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# filename = 'Pu-239_coeff_0.154_MT18.pendf'
# filename = 'Pu-239_coeff_0.015_MT18.pendf'
#
# f = open(filename)
# lines = f.readlines()
# section = ENDF6.find_section(lines, MF=3, MT=18)
# erg, xs = ENDF6.read_table(section)
#
# plt.figure()
# plt.plot(erg, xs, label = 'Data')
# plt.legend()
# plt.grid()
# plt.xlabel('Energy / eV')
# plt.ylabel('$\sigma_{n,f}$ / b')
# plt.xscale('log')
# plt.yscale('log')
# plt.title('Perturbed Pu-239 (n,f) cross sections')
# plt.show()

dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/dataprocessing/fission_perturbations/parallel_processing/pendf_perturbed'
pendf_names = os.listdir(dir)

length_list = []
# def process_file(file):
# 	f = open(f'{dir}/{file}')
# 	lines = f.readlines()
# 	section = ENDF6.find_section(lines, MF=3, MT=18)
# 	erg, xs = ENDF6.read_table(section)
# 	# return(len(erg))
# 	# length_list.append(len(erg))
#
# # with ThreadPoolExecutor(max_workers=1) as executor:
# 	futures = {executor.submit(process_file, file): file for file in pendf_names}
# 	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
# 		result = future.result()
# 		length_list.append(result)


for filename in tqdm.tqdm(pendf_names, total=len(pendf_names)):
	f = open(f'{dir}/{filename}')
	lines = f.readlines()
	section = ENDF6.find_section(lines, MF=3, MT=18)
	erg, xs = ENDF6.read_table(section)

	name_split = filename.split('_')
	coefficient = float(name_split[2])
	coeff_list = [coefficient for i in xs]

	df_temp = pd.DataFrame({'ERG': erg, 'XS': xs, 'P':coeff_list})
	df_temp.to_csv(f'Pu239_{coefficient}_flat_MT18_XS.csv')