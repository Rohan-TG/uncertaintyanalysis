import endf
import os
import matplotlib.pyplot as plt
import numpy as np

file = ('/Users/rntg/PycharmProjects/uncertaintyanalysis/feature-engineering/94239.endf6')

mat = endf.Material(file)


def read_mf5(endf6_file, MT=18):
	for i, set in enumerate(mat.section_data[5, 18]['subsections'][0]['distribution']['g']):
		energies = mat.section_data[5, MT]['subsections'][0]['distribution']['g'][i].x
		y_values = mat.section_data[5, MT]['subsections'][0]['distribution']['g'][i].y

incident_energies = mat.section_data[5, 18]['subsections'][0]['distribution']['E']

y_matrix = []
for i, set in enumerate(mat.section_data[5, 18]['subsections'][0]['distribution']['g']):
	y_values = mat.section_data[5, 18]['subsections'][0]['distribution']['g'][i].y
	y_matrix.append(y_values)

	x = mat.section_data[5, 18]['subsections'][0]['distribution']['g'][i].x


# Plotting
facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(incident_energies)))
ax = plt.figure().add_subplot(projection='3d')
for i, l in enumerate(incident_energies):
	ax.fill_between(x, l, y_matrix[i], x, l, 0, facecolors=facecolors[i], alpha=.7)
ax.set_xlabel('Outbound E / eV')
ax.set_ylabel('Incident E / eV')
ax.set_zlabel('Probability')
plt.title('Pu-239 ENDF/B-VIII.0 PFNS')
plt.savefig('PFNS_example_Pu-239.png', dpi=300)
plt.show()

