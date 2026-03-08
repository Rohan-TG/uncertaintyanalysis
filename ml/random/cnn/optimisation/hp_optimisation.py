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
import keras_tuner as kt


print('\n')
data_directory = input('Data directory: ')
test_data_directory = input('\nTest data directory (x for set to val): ')


data_processes = 5

all_parquets = os.listdir(data_directory)

training_fraction = float(input('\nEnter training data fraction: '))
lower_energy_bound = float(input('\nEnter lower energy bound in eV: '))
patience = int(input('\nPatience: '))

try:
	mask = float(input('\nMask (x skip): '))
except:
	mask = 'x'
	print('Skip masking...')

n_training_samples = int(training_fraction * len(all_parquets))




print('\nFetching training data...')





def fetch_data(datafile, data_dir=data_directory):

	temp_df = pd.read_parquet(f'{data_dir}/{datafile}', engine='pyarrow')
	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]

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

	# xsobject = pu9_mt2xs + pu9_mt4xs + pu9_mt16xs + pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt16xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt16xs + pu1_mt18xs + pu1_mt102xs
	#
	# XS_obj = xsobject

	XS_obj = [pu9_mt2xs, pu9_mt4xs, pu9_mt16xs, pu9_mt18xs, pu9_mt102xs,
				pu0_mt2xs, pu0_mt4xs, pu0_mt16xs, pu0_mt18xs, pu0_mt102xs,
				pu1_mt2xs, pu1_mt4xs, pu1_mt16xs, pu1_mt18xs, pu1_mt102xs,]

	return(XS_obj, keff_value)



# def interpolate_to_default_grid(XS_matrix):
# 	"""Interpolates any XS_matrix data to the PCHIP-sampled Pu-239 MT=18 grid"""
# 	default_df = pd.read_parquet(f'{data_directory}/{all_parquets[0]}')
# 	default_grid = default_df['ERG'].values
# 	default_grid = [e for e in default_grid if e >= lower_energy_bound]
#
# 	native_df = pd.read_parquet(f'{test_data_directory}/{val_files[0]}')
# 	native_grid = native_df['ERG'].values
# 	native_grid = [e for e in native_grid if e >= lower_energy_bound]
#
# 	thinned_XS_matrix = []
# 	for sample in tqdm.tqdm(XS_matrix, total=len(XS_matrix)):
# 		# transposed_sample = sample.transpose()
# 		interpolated_sample = []
# 		for channel_xs in sample:
# 			thinned_xs = np.interp(default_grid, native_grid, channel_xs)
# 			interpolated_sample.append(thinned_xs)
# 		interpolated_sample = np.array(interpolated_sample)
#
# 		thinned_XS_matrix.append(interpolated_sample)
#
# 	thinned_XS_matrix = np.array(thinned_XS_matrix)
#
# 	return thinned_XS_matrix


f =0
training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_parquets)
	if choice not in training_files:
		training_files.append(choice)

val_files = []
for file in all_parquets:
	if file not in training_files:
		val_files.append(file)

keff_train = []  # k_eff labels
XS_train = []

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, train_file) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, keff_value = future.result()
		XS_train.append(xs_values)
		keff_train.append(keff_value)

XS_train = np.array(XS_train)  # shape (num_samples, num_channels, points per channel)

keff_train_mean = np.mean(keff_train)
keff_train_std = np.std(keff_train)
y_train = zscore(keff_train)

print('\nFetching val data...')

XS_val = []
keff_val = []

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, val_file) for val_file in val_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, keff_value_val = future.result()
		XS_val.append(xs_values_val)
		keff_val.append(keff_value_val)

XS_val = np.array(XS_val)
y_val = (np.array(keff_val) - keff_train_mean) / keff_train_std

print('\nScaling all data...')

if test_data_directory == 'x':
	XS_test = XS_val
	y_test = y_val


def process_data(XS_train, XS_val, XS_test):

	channel_matrix_train = [[] for i in range(len(XS_train[0]))] # each element is a matrix of only one channel, e.g. channel_matrix[0] is all the lists containing
	channel_matrix_val = [[] for i in range(len(XS_val[0]))]
	channel_matrix_test = [[] for i in range(len(XS_test[0]))]


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


X_train, X_val, X_test = process_data(XS_train, XS_val, XS_test)

X_test[np.isinf(X_test)] = -1

if type(mask) != str:
	X_test[np.abs(X_test) >= mask] = 0



# le_bound_index = 1 # filters out NaNs

trainstart = time.time()





def build_model(hp):


	#### Begin model stuff



	hp_units = hp.Int('filters', min_value=16, max_value=256, step=16)
	model = keras.Sequential()
	model.add(keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
	model.add(keras.layers.Conv1D(filters=hp_units, kernel_size=2, padding='same', activation='relu',))
	model.add(keras.layers.Flatten())

	n_dense = hp.Int("n_dense_layers", min_value=1, max_value=12, step=1) # varies number of layers used
	for n_layers in range(n_dense):
		node_units = hp.Int(f'dense_{n_layers}_units', min_value=16, max_value=1500, step=50)
		model.add(keras.layers.Dense(node_units, activation='relu'))

	model.add(keras.layers.Dense(1, activation='linear'))
	model.compile(loss='MeanSquaredError', optimizer='adam')

	return model


callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 patience=patience,
										 mode='min',
										 start_from_epoch=3,
										 restore_best_weights=True)

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=100,
    directory='my_tuner_dir',
    project_name='CNN_tuning')

tuner.search(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=2000,
    batch_size=32,
	callbacks = callback
)

import datetime
train_end = time.time()
print(f'Optimisation completed in {datetime.timedelta(seconds=(train_end - trainstart))}')

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

predictions = best_model.predict(X_test)
predictions = predictions.ravel()


rescaled_predictions = []
predictions_list = predictions.tolist()

for pred in predictions_list:
	descaled_p = pred * keff_train_std + keff_train_mean
	rescaled_predictions.append(float(descaled_p))

if test_data_directory == 'x':
	keff_test = keff_val
errors = []
for predicted, true in zip(rescaled_predictions, keff_test):
	errors.append((predicted - true) * 1e5)
	print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

sorted_errors = sorted(errors)
absolute_errors = [abs(x) for x in sorted_errors]
print(f'Average absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')

print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')


print(f"Smallest absolute error: {min(absolute_errors)} pcm")
acceptable_predictions = []
borderline_predictions = []
twenty_pcm_predictions = []
for x in absolute_errors:
	if x <= 5.0:
		acceptable_predictions.append(x)
	if x <= 10.0:
		borderline_predictions.append(x)
	if x <= 20.0:
		twenty_pcm_predictions.append(x)


print(f' {len(acceptable_predictions)} ({len(acceptable_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 5 pcm error')
print(f' {len(borderline_predictions)} ({len(borderline_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 10 pcm error')
print(f' {len(twenty_pcm_predictions)} ({len(twenty_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 20 pcm error)')

import matplotlib.pyplot as plt
save_histogram = input('Save histogram? (y): ')
if save_histogram == 'y':

	histogram_title = input('Histogram title: ')
	scatterplot_title = input('Scatterplot title: ')

	plt.figure()
	plt.hist(sorted_errors, bins=25)
	plt.grid()
	plt.title('Distribution of errors')
	plt.xlabel('Error / pcm')
	plt.ylabel('Count')
	plt.savefig(f'{histogram_title}.png')
	plt.show()

	plt.figure()
	plt.plot(keff_test, errors, 'x')
	plt.grid()
	plt.title('Distribution of errors')
	plt.xlabel('True k_eff')
	plt.ylabel('Error / pcm')
	plt.savefig(f'{scatterplot_title}.png')
	plt.show()

skew_positive = []
skew_negative = []

for x in errors:
	if x >0:
		skew_positive.append(x)
	else:
		skew_negative.append(x)

print(f'skew_positive: {len(skew_positive)} / {len(errors)}')
print(f'skew_negative: {len(skew_negative)} / {len(errors)}')

## Begin error analysis


channel_keys = {'94239_MT2_XS': 0,'94239_MT4_XS': 1,'94239_MT16_XS': 2,
				'94239_MT18_XS': 3,'94239_MT102_XS': 4,'94240_MT2_XS': 5,
				'94240_MT4_XS': 6,'94240_MT16_XS': 7,'94240_MT18_XS': 8,
				'94240_MT102_XS': 9,'94241_MT2_XS': 10,'94241_MT4_XS': 11,
				'94241_MT16_XS': 12,'94241_MT18_XS': 13,'94241_MT102_XS': 14,}
