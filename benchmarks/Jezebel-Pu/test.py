import h5py

import h5py

# Open the file in read-only mode
with h5py.File('statepoint.3050.h5', 'r') as f:
	# List all groups
	print("Keys: %s" % f.keys())
	group = list(f.keys())[0]

	# Access a dataset
	data = f['k_combined']
	print(data)
	allk = data[:]
	print(allk)