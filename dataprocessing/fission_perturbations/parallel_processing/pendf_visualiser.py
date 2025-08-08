import matplotlib.pyplot as plt
from dataprocessing import ENDF6
import os
import tqdm


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
for file in tqdm.tqdm(pendf_names, total=len(pendf_names)):
	f = open(file)
	lines = f.readlines()
	section = ENDF6.find_section(lines, MF=3, MT=18)
	erg, xs = ENDF6.read_table(section)
	length_list.append(len(erg))