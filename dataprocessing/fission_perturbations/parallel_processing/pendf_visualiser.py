import matplotlib.pyplot as plt
from dataprocessing import ENDF6


def plot_data(filename):

	f = open(filename)
	lines = f.readlines()
	section = ENDF6.find_section(lines, MF=3, MT=18)
	erg, xs = ENDF6.read_table(section)

	plt.figure()
	plt.plot(erg, xs, label = 'Data')
	plt.legend()
	plt.grid()
	plt.xlabel('Energy / eV')
	plt.ylabel('$\sigma_{n,f}$ / b')
	plt.title('Perturbed Pu-239 (n,f) cross sections')
	plt.show()