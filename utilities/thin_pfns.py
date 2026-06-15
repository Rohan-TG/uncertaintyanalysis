import os
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie' or computer == 'bethe':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from groupEnergies import pfns_outbound_energies

isotope = input('Isotope: ')
original_directory = input('\nOriginal directory: ')
original_files = os.listdir(original_directory)

new_directory = input('\nTarget: ')
tolerance = float(input('\nTolerance: '))







def thin_relative_error_logx(x, y, rel_tol=tolerance, y_floor=None, max_points=None):
	x = np.asarray(x)
	y = np.asarray(y)
	idx = np.argsort(x)
	x = x[idx]
	y = y[idx]

	# log-x coordinate
	u = np.log(x)

	if y_floor is None:
		y_floor = 1e-12 * np.nanmax(y)

	# Keep array
	keep = np.zeros_like(y, dtype=bool)
	keep[0] = True
	keep[-1] = True

	# Stack of segments as (i, j)
	stack = [(0, len(y)-1)]

	while stack:
		i, j = stack.pop()
		if j <= i + 1:
			continue

		# Build interpolant from endpoints only
		interp = PchipInterpolator(u[[i, j]], y[[i, j]], extrapolate=False)

		k = np.arange(i+1, j)
		yhat = interp(u[k])

		denom = np.maximum(np.abs(y[k]), y_floor)
		err = np.abs(yhat - y[k]) / denom

		m = np.argmax(err)
		if err[m] > rel_tol:
			km = k[m]
			keep[km] = True

			# Split and continue
			stack.append((i, km))
			stack.append((km, j))

			# stop if points budget expended
			if max_points is not None and keep.sum() >= max_points:
				break

	# Return subset in original order
	kept_idx_sorted = np.nonzero(keep)[0]
	kept_idx = idx[kept_idx_sorted]  # indices into original arrays
	return kept_idx, x[kept_idx_sorted], y[kept_idx_sorted]

pfns_original_energies_df = pd.read_parquet(f'{original_directory}/{original_files[0]}')
pfns_original_energies = pfns_outbound_energies
pfns_original = pfns_original_energies_df[1.000000e-05].values



def thin_single_sample(file):

	filename = file.split('.')[0]
	dataframe = pd.read_parquet(f'{original_directory}/{file}')

	incident_energy_columns = dataframe.columns
	thinned_values = []
	for column in dataframe:
		original_pfns = dataframe[column].values
		kept_idx, thinned_energy, thinned_pfns = thin_relative_error_logx(x=pfns_original_energies,
																		  y=original_pfns,
																		  rel_tol=tolerance, )
		thinned_values.append(thinned_pfns)
	saved_energies_df= pd.DataFrame(thinned_energy)
	saved_energies_df.to_csv(f'Saved_MF5_energies_tolerance_{tolerance}_{isotope}.csv')

	new_df = pd.DataFrame(thinned_values, columns=incident_energy_columns)
	new_df.to_parquet(f'{new_directory}/{filename}_tolerance_{tolerance}.parquet')

thin_single_sample(original_files[0])