import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

original_directory = input('\nOriginal directory: ')
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
