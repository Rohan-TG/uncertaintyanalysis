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




lambdas = list(range(1,24))
y_matrix = []
for i, set in enumerate(mat.section_data[5, 18]['subsections'][0]['distribution']['g']):
	y_values = mat.section_data[5, 18]['subsections'][0]['distribution']['g'][i].y
	y_matrix.append(y_values)

	x = mat.section_data[5, 18]['subsections'][0]['distribution']['g'][i].x



facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(lambdas)))
ax = plt.figure().add_subplot(projection='3d')

for i, l in enumerate(lambdas):
	ax.fill_between(x, l, y_matrix[i], x, l, 0, facecolors=facecolors[i], alpha=.7)



