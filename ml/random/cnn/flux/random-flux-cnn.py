import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# 1D CNN

computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore
import tqdm
import keras
import time
import matplotlib.pyplot as plt



xs_directory = input('XS directory: ')
test_data_directory = input('Test data directory (x for set to val): ')

flux_data_directory = input('Flux data directory: ')


data_processes = int(input('Num. data processors: '))

all_parquets = os.listdir(xs_directory)

training_fraction = float(input('Enter training data fraction: '))
lower_energy_bound = float(input('Enter lower energy bound in eV: '))

n_training_samples = int(training_fraction * len(all_parquets))

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_parquets)
	if choice not in training_files:
		training_files.append(choice)

val_files = []
for file in all_parquets:
	if file not in training_files:
		val_files.append(file)



print('Fetching training data...')





def fetch_data(datafile):

	temp_df = pd.read_parquet(f'{xs_directory}/{datafile}', engine='pyarrow')
	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]

	pu9_index = int(datafile.split('_')[4])
	pu0_index = int(datafile.split('_')[6])
	pu1_index = int(datafile.split('_')[8].split('.')[0])


	# keff_value = float(temp_df['keff'].values[0])

	pu9_mt18xs = temp_df['94239_MT18_XS'].values.tolist()
	pu0_mt18xs = temp_df['94240_MT18_XS'].values.tolist()
	pu1_mt18xs = temp_df['94241_MT18_XS'].values.tolist()

	pu9_mt2xs = temp_df['94239_MT2_XS'].values.tolist()
	pu0_mt2xs = temp_df['94240_MT2_XS'].values.tolist()
	pu1_mt2xs = temp_df['94241_MT2_XS'].values.tolist()

	pu9_mt4xs = temp_df['94239_MT4_XS'].values.tolist()
	pu0_mt4xs = temp_df['94240_MT4_XS'].values.tolist()
	pu1_mt4xs = temp_df['94241_MT4_XS'].values.tolist()

	pu9_mt16xs = temp_df['94239_MT16_XS'].values.tolist()
	pu0_mt16xs = temp_df['94240_MT16_XS'].values.tolist()
	pu1_mt16xs = temp_df['94241_MT16_XS'].values.tolist()

	pu9_mt102xs = temp_df['94239_MT102_XS'].values.tolist()
	pu0_mt102xs = temp_df['94240_MT102_XS'].values.tolist()
	pu1_mt102xs = temp_df['94241_MT102_XS'].values.tolist()

	XS_obj = [pu9_mt2xs, pu9_mt4xs, pu9_mt16xs, pu9_mt18xs, pu9_mt102xs,
				pu0_mt2xs, pu0_mt4xs, pu0_mt16xs, pu0_mt18xs, pu0_mt102xs,
				pu1_mt2xs, pu1_mt4xs, pu1_mt16xs, pu1_mt18xs, pu1_mt102xs,]


	# Now fetch spectrum data labels

	flux_file = f'Flux_data_Pu-239_{pu9_index}_Pu-240_{pu0_index}_Pu-241_{pu1_index}.parquet'
	flux_read_obj = pd.read_parquet(f'{flux_data_directory}/{flux_file}', engine='pyarrow')
	flux_data = flux_read_obj['flux'].values
	global flux_lower_bounds, flux_upper_bounds
	flux_lower_bounds = flux_read_obj['low_erg_bounds'].values
	flux_upper_bounds = flux_read_obj['high_erg_bounds'].values

	return(XS_obj,
		   flux_data
		   )



flux_train = [] # flux labels
XS_train = []

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, train_file) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, flux_values = future.result()
		XS_train.append(xs_values)
		flux_train.append(flux_values)

XS_train = np.array(XS_train) # shape (num_samples, num_channels, points per channel)
flux_train = np.array(flux_train)


def scale_flux(flux_array, train_mode, means = None, stds = None):
	"""setting train_mode to True just makes this function return the means and stds. Otherwise not returned"""

	normalised_flux_array = []
	for flux_set in flux_array:


	transposed_flux_array = flux_array.transpose()
	if train_mode:
		scaling_columns = []
		scaling_column_means = []
		scaling_column_stds = []
		for energy_column in transposed_flux_array:
			scaling_columns.append(zscore(energy_column))
			scaling_column_means.append(np.mean(energy_column))
			scaling_column_stds.append(np.std(energy_column))

		return scaling_columns, scaling_column_means, scaling_column_stds
	else:
		scaling_columns = []
		scaling_column_means = []
		scaling_column_stds = []
		for energy_column, mean, std in zip(transposed_flux_array, means, stds):
			scaling_columns.append((np.array(energy_column) - mean) / std)
		return scaling_columns

y_train, scaling_means_train, scaling_stds_train = scale_flux(flux_train, train_mode=True)

XS_val = []
flux_val = []
print('Fetching val data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, val_file) for val_file in val_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, flux_value_val = future.result()
		XS_val.append(xs_values_val)
		flux_val.append(flux_value_val)

XS_val = np.array(XS_val)

y_val = scale_flux(flux_val, train_mode=False)


if test_data_directory != 'x':
	print('Fetching test data...')
	test_files = os.listdir(test_data_directory)

	XS_test = []
	keff_test = []
	with ProcessPoolExecutor(max_workers=data_processes) as executor:
		futures_test = [executor.submit(fetch_data, test_file) for test_file in test_files]

		for future_test in tqdm.tqdm(as_completed(futures_test), total=len(futures_test)):
			xs_values_test, keff_value_test = future_test.result()
			XS_test.append(xs_values_test)
			keff_test.append(keff_value_test)

	XS_test = np.array(XS_test)
	y_test = (np.array(keff_test) - keff_train_mean) / keff_train_std


print('Scaling training data...')


# le_bound_index = 1 # filters out NaNs

def process_data(XS_train, XS_val, XS_test):


	channel_matrix_train = [[] for i in range(len(XS_train[0]))] # each element is a matrix of only one channel, e.g. channel_matrix[0] is all the lists containing
	channel_matrix_val = [[] for i in range(len(XS_val[0]))]
	channel_matrix_test = [[] for i in range(len(XS_test[0]))]


	# Pu-239 (n,el)
	scaled_channel_matrix_train = []
	scaled_channel_matrix_val = []
	scaled_channel_matrix_test = []


	for matrix in tqdm.tqdm(XS_train, total =len(XS_train)):
		# Each matrix has shape (num channels, points per channel)
		for channel_index, channel in enumerate(matrix):
			channel_matrix_train[channel_index].append(channel)

		# channel_matrix now has shape (num channels, num samples, points per channel)
		# Each element of channel matrix has shape (num samples, points per channel)
	for matrix in tqdm.tqdm(XS_val, total =len(XS_val)):
		for channel_index, channel in enumerate(matrix):
			channel_matrix_val[channel_index].append(channel)

	for matrix in tqdm.tqdm(XS_test, total =len(XS_test)):
		for channel_index, channel in enumerate(matrix):
			channel_matrix_test[channel_index].append(channel)

	#################################################################################################################################################################
	for channel_data_train, channel_data_val, channel_data_test in zip(channel_matrix_train, channel_matrix_val, channel_matrix_test): # each iterative variable is the tensor of one specific channel e.g. Pu-239 fission, for all samples
		transposed_matrix_train = np.transpose(channel_data_train) # shape (energy points per sample, num samples)
		transposed_matrix_val = np.transpose(channel_data_val)
		transposed_matrix_test = np.transpose(channel_data_test)

		transposed_scaled_channel_train = []
		transposed_scaled_channel_val = []
		transposed_scaled_channel_test = []
		for energy_point_train, energy_point_val, energy_point_test in zip(transposed_matrix_train[:-1], transposed_matrix_val[:-1], transposed_matrix_test[:-1]): # each point on the unionised energy grid

			train_mean = np.mean(energy_point_train)
			train_std = np.std(energy_point_train)

			scaled_point_train = zscore(energy_point_train)
			transposed_scaled_channel_train.append(scaled_point_train)

			scaled_point_val = (np.array(energy_point_val) - train_mean) / train_std
			transposed_scaled_channel_val.append(scaled_point_val)

			scaled_point_test = (np.array(energy_point_test) - train_mean) / train_std
			transposed_scaled_channel_test.append(scaled_point_test)

		scaled_channel_train = np.array(transposed_scaled_channel_train)
		scaled_channel_train = scaled_channel_train.transpose()
		scaled_channel_matrix_train.append(scaled_channel_train)

		scaled_channel_val = np.array(transposed_scaled_channel_val)
		scaled_channel_val = scaled_channel_val.transpose()
		scaled_channel_matrix_val.append(scaled_channel_val)

		scaled_channel_test = np.array(transposed_scaled_channel_test)
		scaled_channel_test = scaled_channel_test.transpose()
		scaled_channel_matrix_test.append(scaled_channel_test)

	###################################################################################################################################################################
	# print('Forming scaled training data...')
	X_matrix_train = [[] for i in range(XS_train.shape[0])] # number of samples
	X_matrix_val = [[] for i in range(XS_val.shape[0])]
	X_matrix_test = [[] for i in range(XS_test.shape[0])]

	for scaled_observable in scaled_channel_matrix_train:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_train[sample_index].append(channel_sample)

	for scaled_observable in scaled_channel_matrix_val:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_val[sample_index].append(channel_sample)

	for scaled_observable in scaled_channel_matrix_test:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_test[sample_index].append(channel_sample)

	X_matrix_train = np.array(X_matrix_train)
	X_matrix_train[np.isnan(X_matrix_train)] = 0

	X_matrix_val = np.array(X_matrix_val)
	X_matrix_val[np.isnan(X_matrix_val)] = 0 # changes nans to 0

	X_matrix_test = np.array(X_matrix_test)
	X_matrix_test[np.isnan(X_matrix_test)] = 0

	return X_matrix_train, X_matrix_val, X_matrix_test


if test_data_directory == 'x':
	XS_test = XS_val
	y_test = y_val

X_train, X_val, X_test = process_data(XS_train, XS_val, XS_test)


#
# callback = keras.callbacks.EarlyStopping(monitor='val_loss',
# 										 # min_delta=0.005,
# 										 patience=50,
# 										 mode='min',
# 										 start_from_epoch=3,
# 										 restore_best_weights=True)




# model = keras.Sequential()
# model.add(keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
# model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(900, activation='relu'))
# model.add(keras.layers.Dense(750, activation='relu'))
# model.add(keras.layers.Dense(550, activation='relu'))
# model.add(keras.layers.Dense(400, activation='relu'))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(200, activation='relu'))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(1, activation='linear'))
# model.compile(loss='MeanSquaredError', optimizer='adam')
#
#
# import datetime
# trainstart = time.time()
# history = model.fit(X_train,
# 					y_train,
# 					epochs=1000,
# 					batch_size=32,
# 					callbacks=callback,
# 					validation_data=(X_val, y_val),
# 					verbose=1)
#
# train_end = time.time()
# print(f'Training completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
# predictions = model.predict(X_val)
# predictions = predictions.ravel()
