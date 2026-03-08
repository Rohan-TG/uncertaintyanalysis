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



print('\n')
data_directory = input('Data directory: ')
test_data_directory = input('\nTest data directory (x for set to val): ')
scale_separately = input('\nScale test with separate statistics? (y): ')
if scale_separately == 'y':
	scale_separately = True
else:
	scale_separately = False

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

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_parquets)
	if choice not in training_files:
		training_files.append(choice)

val_files = []
for file in all_parquets:
	if file not in training_files:
		val_files.append(file)



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



keff_train = [] # k_eff labels
XS_train = []

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, train_file) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, keff_value = future.result()
		XS_train.append(xs_values)
		keff_train.append(keff_value)

XS_train = np.array(XS_train) # shape (num_samples, num_channels, points per channel)

keff_train_mean = np.mean(keff_train)
keff_train_std = np.std(keff_train)
y_train = zscore(keff_train)

XS_val = []
keff_val = []

print('\nFetching val data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, val_file) for val_file in val_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, keff_value_val = future.result()
		XS_val.append(xs_values_val)
		keff_val.append(keff_value_val)

XS_val = np.array(XS_val)
y_val = (np.array(keff_val) - keff_train_mean) / keff_train_std

def interpolate_to_default_grid(XS_matrix):
	"""Interpolates any XS_matrix data to the PCHIP-sampled Pu-239 MT=18 grid"""
	default_df = pd.read_parquet(f'{data_directory}/{all_parquets[0]}')
	default_grid = default_df['ERG'].values
	default_grid = [e for e in default_grid if e >= lower_energy_bound]

	native_df = pd.read_parquet(f'{test_data_directory}/{test_files[0]}')
	native_grid = native_df['ERG'].values
	native_grid = [e for e in native_grid if e >= lower_energy_bound]

	thinned_XS_matrix = []
	for sample in tqdm.tqdm(XS_matrix, total=len(XS_matrix)):
		# transposed_sample = sample.transpose()
		interpolated_sample = []
		for channel_xs in sample:
			thinned_xs = np.interp(default_grid, native_grid, channel_xs)
			interpolated_sample.append(thinned_xs)
		interpolated_sample = np.array(interpolated_sample)

		thinned_XS_matrix.append(interpolated_sample)

	thinned_XS_matrix = np.array(thinned_XS_matrix)

	return thinned_XS_matrix



if test_data_directory != 'x':
	print('\nFetching test data...')
	test_files = os.listdir(test_data_directory)

	raw_XS_test = []
	keff_test = []
	with ProcessPoolExecutor(max_workers=data_processes) as executor:
		futures_test = [executor.submit(fetch_data, test_file, test_data_directory) for test_file in test_files]

		for future_test in tqdm.tqdm(as_completed(futures_test), total=len(futures_test)):
			xs_values_test, keff_value_test = future_test.result()
			raw_XS_test.append(xs_values_test)
			keff_test.append(keff_value_test)

	print('Processing test data...')
	temp_XS_test = np.array(raw_XS_test)
	XS_test = interpolate_to_default_grid(temp_XS_test)
	y_test = (np.array(keff_test) - keff_train_mean) / keff_train_std


print('\nScaling all data...')


# le_bound_index = 1 # filters out NaNs

def process_data(XS_train, XS_val, XS_test, scale_separately = False):

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
	if not scale_separately:
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
	else:
		# Scale training and val together
		for channel_data_train, channel_data_val in zip(channel_matrix_train, channel_matrix_val): # each iterative variable is the tensor of one specific channel e.g. Pu-239 fission, for all samples
			transposed_matrix_train = np.transpose(channel_data_train) # shape (energy points per sample, num samples)
			transposed_matrix_val = np.transpose(channel_data_val)

			transposed_scaled_channel_train = []
			transposed_scaled_channel_val = []
			for energy_point_train, energy_point_val in zip(transposed_matrix_train[:-1], transposed_matrix_val[:-1]): # each point on the unionised energy grid

				train_mean = np.mean(energy_point_train)
				train_std = np.std(energy_point_train)

				scaled_point_train = zscore(energy_point_train)
				transposed_scaled_channel_train.append(scaled_point_train)

				scaled_point_val = (np.array(energy_point_val) - train_mean) / train_std
				transposed_scaled_channel_val.append(scaled_point_val)

			scaled_channel_train = np.array(transposed_scaled_channel_train)
			scaled_channel_train = scaled_channel_train.transpose()
			scaled_channel_matrix_train.append(scaled_channel_train)

			scaled_channel_val = np.array(transposed_scaled_channel_val)
			scaled_channel_val = scaled_channel_val.transpose()
			scaled_channel_matrix_val.append(scaled_channel_val)

		# scale only test alone
		for channel_data_test in channel_matrix_test:  # each iterative variable is the tensor of one specific channel e.g. Pu-239 fission, for all samples
			transposed_matrix_test = np.transpose(channel_data_test)  # shape (energy points per sample, num samples)

			transposed_scaled_channel_test = []
			for energy_point_test in  transposed_matrix_test[:-1]:  # each point on the unionised energy grid

				scaled_point_test = zscore(energy_point_test)
				transposed_scaled_channel_test.append(scaled_point_test)

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

X_train, X_val, X_test = process_data(XS_train, XS_val, XS_test, scale_separately=scale_separately)

X_test[np.isinf(X_test)] = -1

if type(mask) != str:
	X_test[np.abs(X_test) >= mask] = 0

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=patience,
										 mode='min',
										 start_from_epoch=3,
										 restore_best_weights=True)




print('\n\n')
model = keras.Sequential()
model.add(keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', activation='relu',))
# model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',))
# model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(900, activation='relu'))
model.add(keras.layers.Dense(750, activation='relu'))
model.add(keras.layers.Dense(550, activation='relu'))
model.add(keras.layers.Dense(400, activation='relu'))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='MeanSquaredError', optimizer='adam')


import datetime
trainstart = time.time()
history = model.fit(X_train,
					y_train,
					epochs=1000,
					batch_size=16,
					callbacks=callback,
					validation_data=(X_val, y_val),
					verbose=1)

train_end = time.time()
print(f'Training completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
predictions = model.predict(X_test)
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

### Begin error analysis

original_data = pd.read_parquet('/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/original_data_ENDFBVIII.0.parquet')

def check_ratios(quantity):
	ratio = original_data[quantity] / perturbed_data_file[quantity]
	# average_deviation = np.mean(1-ratio)
	absolute_average_deviation = np.mean(np.abs(1-ratio))
	return absolute_average_deviation

deviation_dictionary = {'94239_MT18_XS': [],'94239_MT2_XS': [],'94240_MT18_XS': [],'94240_MT2_XS': [],'94241_MT18_XS':[],'94241_MT2_XS': []}

for error, data_file in zip(errors, val_files):
	perturbed_data_file = pd.read_parquet(f'{data_directory}/{data_file}')

	deviation_dictionary['94239_MT18_XS'].append(check_ratios(quantity='94239_MT18_XS'))
	deviation_dictionary['94239_MT2_XS'].append(check_ratios(quantity='94239_MT2_XS'))

	deviation_dictionary['94240_MT18_XS'].append(check_ratios(quantity='94240_MT18_XS'))
	deviation_dictionary['94240_MT2_XS'].append(check_ratios(quantity='94240_MT2_XS'))

	deviation_dictionary['94241_MT18_XS'].append(check_ratios(quantity='94241_MT18_XS'))
	deviation_dictionary['94241_MT2_XS'].append(check_ratios(quantity='94241_MT2_XS'))
		# print(f'\nML Error: {error:0.0f} pcm, Absolute Pu-239 MT18 deviation: {100* absolute_average_deviation:0.1f} \n %')

# figure_name = input('Deviation figure name: ')
def plot_deviation_analysis(deviations, quantity):

	plt.figure()
	plt.plot(deviations, errors, 'x')
	plt.grid()
	plt.title(f'Average {quantity} perturbation level vs. error')
	plt.xlabel('Absolute deviation')
	plt.ylabel('Error / pcm')
	plt.savefig(f'{quantity}_deviation_analysis.png')

for key in deviation_dictionary:
	plot_deviation_analysis(deviations=deviation_dictionary[key], quantity=key)
#######################################################################################################################

more_analysis = input('More analysis (1): ')
if more_analysis == '1':
	acceptable_batch = []
	acceptable_true = []

	unacceptable_batch = []
	unacceptable_true = []

	for mlerror, truevalue in zip(errors, keff_test):
		if abs(mlerror) > 10.0:
			unacceptable_batch.append(mlerror)
			unacceptable_true.append(truevalue)
		else:
			acceptable_batch.append(mlerror)
			acceptable_true.append(truevalue)

	plt.figure()
	plt.plot(unacceptable_true, unacceptable_batch, 'x', color='red', label = 'unacceptable')
	plt.plot(acceptable_true, acceptable_batch, 'x', color='blue', label = 'acceptable')
	plt.grid()
	plt.legend()
	plt.xlabel('True k_eff')
	plt.ylabel('Error / pcm')
	plt.savefig('Comparison_of_acceptable_and_unacceptable.png')




	keff_sections = np.arange(0.97, 1.06, 0.005)
	step_size = 0.005

	bin_array = []
	for border in keff_sections:
		bin_array.append([border, border + step_size])
	# defines the bins for the bar chart

	error_groups_sorted_by_bin = [[] for g in bin_array]
	true_value_groups_sorted_by_bin = [[] for g in bin_array]
	for group_index, group in enumerate(bin_array):
		for error, true_value in zip(errors, keff_test):
			if true_value >= group[0] and true_value <= group[1]:
				error_groups_sorted_by_bin[group_index].append(error)
				true_value_groups_sorted_by_bin[group_index].append(true_value)



	mean_error_by_keff_group = []
	for error_grouping in error_groups_sorted_by_bin:
		mean_error_by_keff_group.append(np.mean(np.abs(error_grouping)))

	plt.figure()
	plt.plot(bin_array, mean_error_by_keff_group, 'x-', color='red', label = 'mean error')
	plt.grid()
	plt.ylabel('Mean error / pcm')
	plt.xlabel('True k_eff')
	plt.legend()
	plt.savefig('group_error_plot.png')



	##### Plotting error vs. signal amplitude and variance

	signal_mean = []
	signal_variance = []
	for signal_input in X_test:
		signal_mean.append(np.mean(signal_input))
		signal_variance.append(np.var(signal_input))

	plt.figure()
	plt.plot(signal_mean, errors, 'x', color='red')
	plt.grid()
	plt.ylabel('Mean error / pcm')
	plt.xlabel('Signal mean')
	plt.legend()
	plt.savefig('signal_mean.png')

	plt.figure()
	plt.plot(signal_variance, errors, 'x', color='blue')
	plt.xlabel('Signal variance')
	plt.ylabel('Mean error / pcm')
	plt.legend()
	plt.grid()
	plt.savefig('signal_variance.png')


def analyse_channel(channel_idx):
	sample_mean = []
	sample_variance = []
	for sample in X_test:
		sample_mean.append(np.mean(sample[channel_idx]))
		sample_variance.append(np.var(sample[channel_idx]))

	return sample_mean, sample_variance

def plot_means_and_vars(means, variances, channel_name):
	plt.figure()
	plt.plot(means, errors, 'x', color='red')
	plt.xlabel('Channel means')
	plt.title('error as function of mean of '+ channel_name)
	plt.ylabel('Error / pcm')
	plt.grid()
	plt.savefig(f'{channel_name}_means.png')

	plt.figure()
	plt.plot(variances, errors, 'x', color='blue')
	plt.xlabel('Channel variances')
	plt.title('error as function of variance of '+ channel_name)
	plt.ylabel('Error / pcm')
	plt.grid()
	plt.savefig(f'{channel_name}_variances.png')