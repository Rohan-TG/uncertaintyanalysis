import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# 1D CNN

computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
	flux_data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/random/pu-only/all-channels/flux_data'
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
patience = int(input('Patience: '))
# flux_data_directory = input('Flux data directory: ')


# data_processes = int(input('Num. data processors: '))
data_processes = 6
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

	keff_value = float(temp_df['keff'].values[0])

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

	# xsobject = pu9_mt2xs + pu9_mt4xs +  pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt18xs + pu1_mt102xs
	xsobject = pu9_mt2xs + pu9_mt4xs + pu9_mt16xs + pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt16xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt16xs + pu1_mt18xs + pu1_mt102xs

	XS_obj = xsobject


	# Now fetch spectrum data labels

	flux_file = f'Flux_data_Pu-239_{pu9_index}_Pu-240_{pu0_index}_Pu-241_{pu1_index}.parquet'
	flux_read_obj = pd.read_parquet(f'{flux_data_directory}/{flux_file}', engine='pyarrow')
	flux_data = flux_read_obj['flux'].values
	flux_error = flux_read_obj['flux_errror']

	global flux_lower_bounds, flux_upper_bounds
	flux_lower_bounds = flux_read_obj['low_erg_bounds'].values
	flux_upper_bounds = flux_read_obj['high_erg_bounds'].values

	return(XS_obj,
		   flux_data,
		   flux_error,
		   )




flux_train = [] # flux labels
flux_train_error = []
XS_train = []


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, train_file) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, flux_values, flux_error = future.result()
		XS_train.append(xs_values)
		flux_train.append(flux_values)
		flux_train_error.append(flux_error)

XS_train = np.array(XS_train) # shape (num_samples, num_channels, points per channel)
flux_train = np.array(flux_train) # matrix
flux_train_error = np.array(flux_train_error)


def scale_flux(flux_array, flux_error_array, train_mode = False, means = None, stds = None):
	"""setting train_mode to True just makes this function return the means and stds. Otherwise not returned"""

	normalised_flux_array = []
	normalised_flux_error_array = []
	for flux_set, flux_error_set in zip(flux_array, flux_error_array):
		area = np.sum(flux_set)
		norm_flux_vector = np.array(flux_set) / area
		norm_flux_error = np.array(flux_error_set) / area
		normalised_flux_array.append(norm_flux_vector)
		normalised_flux_error_array.append(norm_flux_error)

	normalised_flux_array = np.array(normalised_flux_array)
	normalised_flux_error_array = np.array(normalised_flux_error_array)

	transposed_flux_array = normalised_flux_array.transpose()
	if train_mode:
		scaling_columns = []
		scaling_column_means = []
		scaling_column_stds = []
		for energy_column in transposed_flux_array:
			scaling_columns.append(zscore(energy_column))
			scaling_column_means.append(np.mean(energy_column))
			scaling_column_stds.append(np.std(energy_column))

		scaling_column_stds = np.array(scaling_column_stds)
		scaling_column_means = np.array(scaling_column_means)
		scaled_transposed_flux_array = np.array(scaling_columns)
		scaled_flux_array = scaled_transposed_flux_array.transpose()
		return scaled_flux_array, scaling_column_means, scaling_column_stds, normalised_flux_error_array

	else:
		scaling_columns = []
		for energy_column, mean, std in zip(transposed_flux_array, means, stds):
			scaling_columns.append((np.array(energy_column) - mean) / std)

		scaled_transposed_flux_array = np.array(scaling_columns)
		scaled_flux_array = scaled_transposed_flux_array.transpose()
		return scaled_flux_array, normalised_flux_error_array
	# return normalised_flux_array

def descaler(scaled_flux_array, means, stds):
	transposed_flux_array = scaled_flux_array.transpose()
	rescaled_flux_array = []
	for energy_column, mean, std in zip(transposed_flux_array, means, stds):
		rescaled_flux_array.append(energy_column * std + mean)

	rescaled_flux_array = np.array(rescaled_flux_array)
	rescaled_flux_array = rescaled_flux_array.transpose()
	rescaled_flux_array = rescaled_flux_array
	return rescaled_flux_array

y_train, scaling_means, scaling_stds, flux_errors_train = scale_flux(flux_train, flux_error_array=flux_train_error, train_mode=True)

XS_val = []
flux_val = []
flux_val_error = []
print('Fetching val data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, val_file) for val_file in val_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, flux_value_val, flux_error_val = future.result()
		XS_val.append(xs_values_val)
		flux_val.append(flux_value_val)
		flux_val_error.append(flux_error_val)

XS_val = np.array(XS_val)

y_val, flux_errors_val = scale_flux(flux_val, flux_error_array=flux_val_error, train_mode=False, means=scaling_means, stds=scaling_stds)
y_val = np.array(y_val)

flux_errors_val = np.array(flux_errors_val)

if test_data_directory != 'x':
	print('Fetching test data...')
	test_files = os.listdir(test_data_directory)

	XS_test = []
	flux_test = []
	flux_test_error = []
	with ProcessPoolExecutor(max_workers=data_processes) as executor:
		futures_test = [executor.submit(fetch_data, test_file) for test_file in test_files]

		for future_test in tqdm.tqdm(as_completed(futures_test), total=len(futures_test)):
			xs_values_test, flux_value_test, flux_test_err = future_test.result()
			XS_test.append(xs_values_test)
			flux_test.append(flux_value_test)
			flux_test_error.append(flux_test_err)

	XS_test = np.array(XS_test)
	y_test, flux_errors_test = scale_flux(flux_test, flux_error_array=flux_test_error, train_mode=False, means=scaling_means, stds=scaling_stds)


print('Scaling training data...')


le_bound_index = 1 # filters out NaNs
def process_data_mlp(XS_train, XS_test):
	xs_train = np.array(XS_train)
	scaling_matrix_xtrain = xs_train.transpose()
	scaled_columns_xtrain = []

	training_column_means = []
	training_column_stds = []

	for column in tqdm.tqdm(scaling_matrix_xtrain[le_bound_index:-1],
							total=len(scaling_matrix_xtrain[le_bound_index:-1])):
		scaled_column = zscore(column)

		column_mean = np.mean(column)
		column_std = np.std(column)
		training_column_means.append(column_mean)
		training_column_stds.append(column_std)
		scaled_columns_xtrain.append(scaled_column)

	scaled_columns_xtrain = np.array(scaled_columns_xtrain)
	X_train = scaled_columns_xtrain.transpose()

	scaling_matrix_xtest = XS_test.transpose()

	scaled_columns_xtest = []
	print('\nScaling test data...')
	for column, c_mean, c_std in tqdm.tqdm(
			zip(scaling_matrix_xtest[le_bound_index:-1], training_column_means, training_column_stds),
			total=len(scaling_matrix_xtest[le_bound_index:-1])):
		# scaled_column = zscore(column)

		scaled_column = (np.array(column) - c_mean) / c_std
		scaled_columns_xtest.append(scaled_column)

	scaled_columns_xtest = np.array(scaled_columns_xtest)
	X_test = scaled_columns_xtest.transpose()

	X_test = np.nan_to_num(X_test, nan=0.0)

	# train_mask = ~np.isnan(X_train).any(axis=0)
	# X_train = X_train[:, train_mask]
	X_train = np.nan_to_num(X_train, nan=0.0)

	return X_train, X_test









if test_data_directory == 'x':
	XS_test = XS_val
	y_test = y_val




callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=patience,
										 mode='min',
										 start_from_epoch=3,
										 restore_best_weights=True)


X_train, X_val = process_data_mlp(XS_train, XS_val)

model = keras.Sequential()
model.add(keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1900, activation='relu'))
model.add(keras.layers.Dense(1750, activation='relu'))
model.add(keras.layers.Dense(1650, activation='relu'))
model.add(keras.layers.Dense(550, activation='relu'))
model.add(keras.layers.Dense(400, activation='relu'))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(y_val.shape[1], activation='linear'))
model.compile(loss='MeanSquaredError', optimizer='adam')


import datetime
trainstart = time.time()
history = model.fit(X_train,
					y_train,
					epochs=1000,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_val, y_val),
					verbose=1)

train_end = time.time()
print(f'Training completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
predictions = model.predict(X_val)

from sklearn.metrics import r2_score
r2s = []

rescaled_full_p = [] # contains predictions
rescaled_y_val = [] # contains labels
pct_list = []
over_limit_list = []
for idx, (p_set, true_set) in enumerate(zip(predictions, y_val)):
	rescaled_predictions = descaler(p_set, means=scaling_means, stds=scaling_stds)
	rescaled_full_p.append(rescaled_predictions)

	rescaled_true_set = descaler(true_set, means=scaling_means, stds=scaling_stds)
	rescaled_y_val.append(rescaled_true_set)

	true_percentage_errors = 100 * (np.array(flux_errors_val[idx]) / np.array(rescaled_true_set))

	ratios = np.array(rescaled_predictions) / np.array(rescaled_true_set)
	pct_deviation = (ratios - 1.0) * 100
	pct_list.append(pct_deviation)

	count = 0
	for point_ml_error, point_real_error in zip(pct_deviation, true_percentage_errors):
		if abs(point_ml_error) > abs(point_real_error * 2):
			count += 1

	over_limit_pct = (count / len(pct_deviation)) * 100
	over_limit_list.append(over_limit_pct)
	print(f'{idx} - {over_limit_pct:0.1f}% Points over limit, Mean: {np.mean(ratios):0.4f} Max: {max(ratios):0.4f} Min: {min(ratios):0.4f} R2: {r2_score(rescaled_predictions, rescaled_true_set):0.5f}')
	r2s.append(r2_score(rescaled_predictions, rescaled_true_set))

print(f'Mean R2: {np.mean(r2s):0.5f}')
print(f'Avg. {np.mean(over_limit_list):0.1f}% +- {np.std(over_limit_list):0.1f}% over limit')

d1, d2, d3 = fetch_data(all_parquets[0])

grid = []
for i in flux_lower_bounds:
	grid.append(float(i))

for j in flux_upper_bounds:
	grid.append(float(j))

grid.sort()
edges = []
for i in grid:
	if i not in edges:
		edges.append(i)
widths = np.diff(edges)
def plot_index(sample_idx):

	true_pct_error = 100 * (np.array(flux_errors_val[sample_idx]) / np.array(rescaled_y_val[sample_idx]))

	sigma_2_upper = 2* true_pct_error
	sigma_2_lower = -2 * true_pct_error



	plt.figure()
	plt.bar(edges[:-1], rescaled_full_p[sample_idx], width=widths, label='Prediction')
	plt.bar(edges[:-1], rescaled_y_val[sample_idx], width=widths, label = "True")
	plt.grid()
	plt.legend()
	plt.xlabel('Energy / MeV')
	plt.ylabel('Normalised flux')
	plt.xscale('log')
	plt.ylim(0,0.015)
	plt.savefig(f'{sample_idx}_bar_mlp.png')


	scale_log = input('Log scale? (y): ')
	plt.figure()
	plt.bar(edges[:-1], pct_list[sample_idx], width=widths, label = 'ML error')
	plt.plot(edges[:-1], true_pct_error, label='MC Error')
	plt.fill_between(edges[:-1], sigma_2_lower, sigma_2_upper, color='r', alpha=0.3, label = '2$\sigma$')
	plt.xlabel('Energy / Mev')
	plt.ylabel('% Deviation')
	if scale_log == 'y':
		plt.xscale('log')
	plt.grid()
	plt.legend()
	plt.savefig(f'{sample_idx}_val_pct_error_bar_mlp.png')