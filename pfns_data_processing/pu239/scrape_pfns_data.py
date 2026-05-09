import endf
import os
import matplotlib.pyplot as plt
import numpy as np

# file = ('/Users/rntg/PycharmProjects/uncertaintyanalysis/feature-engineering/94239.endf6')
master_file = '94239_master_file.ENDFB8_0'
perturbed_endf6 = '94239_4.endf6'

# mat = endf.Material(file)
matp = endf.Material(perturbed_endf6)
mat_original = endf.Material(master_file)

def read_mf5(mat, MT=18):
	if MT != 455:
		for i, set in enumerate(mat.section_data[5, MT]['subsections'][0]['distribution']['g']):
			energies = mat.section_data[5, MT]['subsections'][0]['distribution']['g'][i].x
			y_values = mat.section_data[5, MT]['subsections'][0]['distribution']['g'][i].y

		incident_energies = mat.section_data[5, MT]['subsections'][0]['distribution']['E']

		y_matrix = []
		for i, set in enumerate(mat.section_data[5, MT]['subsections'][0]['distribution']['g']):
			y_values = mat.section_data[5, MT]['subsections'][0]['distribution']['g'][i].y
			y_matrix.append(y_values)

			x = mat.section_data[5, MT]['subsections'][0]['distribution']['g'][i].x

	else:
		energies = mat.section_data[5, MT]['subsections'][0]['distribution']['g'].x
		y_values = mat.section_data[5, MT]['subsections'][0]['distribution']['g'].y

		incident_energies = [1]

		y_matrix = []
		y_values = mat.section_data[5, MT]['subsections'][0]['distribution']['g'].y
		y_matrix.append(y_values)

		x = mat.section_data[5, MT]['subsections'][0]['distribution']['g'].x


	return incident_energies, y_matrix, x

incident_energiesp, y_matrixp, xp = read_mf5(matp)

incident_energies_original, y_matrix_original, x_original = read_mf5(mat_original)


# Plotting
facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(incident_energiesp)))
ax = plt.figure().add_subplot(projection='3d')
for i, l in enumerate(incident_energiesp):
	ax.fill_between(xp, l, y_matrixp[i], xp, l, 0, facecolors=facecolors[i], alpha=.7)
ax.set_xlabel('Outbound E / eV')
ax.set_ylabel('Incident E / eV')
ax.set_zlabel('Probability')
plt.title('Pu-239 ENDF/B-VIII.0 PFNS')
# plt.savefig('PFNS_example_Pu-239.png', dpi=300)
plt.show()

ratio_matrix = []
for p, o in zip(y_matrixp, y_matrix_original):
	ratio_matrix.append(np.array(p) / np.array(o))

plt.figure()
plt.plot(xp[:-1],ratio_matrix[0][:-1])
plt.grid()
plt.show()