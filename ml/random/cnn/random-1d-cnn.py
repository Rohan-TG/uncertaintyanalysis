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




data_directory = input('Data directory: ')
data_processes = int(input('Num. data processors: '))

all_parquets = os.listdir(data_directory)

training_fraction = float(input('Enter training data fraction: '))
lower_energy_bound = float(input('Enter lower energy bound in eV: '))

n_training_samples = int(training_fraction * len(all_parquets))

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_parquets)
	if choice not in training_files:
		training_files.append(choice)

test_files = []
for file in all_parquets:
	if file not in training_files:
		test_files.append(file)

print('Fetching training data...')





def fetch_data(datafile):

	temp_df = pd.read_parquet(f'{data_directory}/{datafile}', engine='pyarrow')
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
y_train = zscore(keff_train)

print('Scaling training data...')


le_bound_index = 1 # filters out NaNs


channel_matrix = [[] for i in range(len(XS_train[0]))] # each element is a matrix of only one channel, e.g. channel_matrix[0] is all the lists containing
# Pu-239 (n,el)
scaled_channel_matrix = [[] for i in range(len(XS_train[0]))]

for matrix in XS_train:
	# Each matrix has shape (num channels, points per channel)
	for channel_index, channel in enumerate(matrix):
		channel_matrix[channel_index].append(channel)

	# channel_matrix now has shape (num channels, num samples, points per channel)
	# Each element of channel matrix has shape (num samples, points per channel)

	for scaling_channel_index, channel_data in enumerate(channel_matrix): # each iterative variable is the matrix of one specific channel e.g. Pu-239 fission
		transposed_matrix = np.transpose(channel_data) # shape (points per sample, num samples)

		transposed_scaled_channel = []
		for energy_point in transposed_matrix: # each point on the unionised energy grid
			scaled_point = zscore(energy_point)
			transposed_scaled_channel.append(scaled_point)

		scaled_channel = np.array(transposed_scaled_channel)
		scaled_channel = scaled_channel.transpose()

		scaled_channel_matrix[scaling_channel_index].append(scaled_channel)


	# for column in tqdm.tqdm(scaling_matrix_xtrain[le_bound_index:-1], total=len(scaling_matrix_xtrain[le_bound_index:-1])):
	# 	scaled_column = zscore(column)
	# 	scaled_columns_xtrain.append(scaled_column)

# scaled_columns_xtrain = np.array(scaled_columns_xtrain)
# X_train = scaled_columns_xtrain.transpose()
#
#
# XS_test = []
# keff_test = []
#
# print('Fetching test data...')
#
#
# with ProcessPoolExecutor(max_workers=data_processes) as executor:
# 	futures = [executor.submit(fetch_data, test_file) for test_file in test_files]
#
# 	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
# 		xs_values_test, keff_value_test = future.result()
# 		XS_test.append(xs_values_test)
# 		keff_test.append(keff_value_test)

# XS_test = np.array(XS_test)
# keff_mean = np.mean(keff_test)
# keff_std = np.std(keff_test)
# y_test = zscore(keff_test)
#
# scaling_matrix_xtest = XS_test.transpose()
#
# scaled_columns_xtest = []
# print('Scaling test data...')
# for column in tqdm.tqdm(scaling_matrix_xtest[le_bound_index:-1], total=len(scaling_matrix_xtest[le_bound_index:-1])):
# 	scaled_column = zscore(column)
# 	scaled_columns_xtest.append(scaled_column)
#
# scaled_columns_xtest = np.array(scaled_columns_xtest)
# X_test = scaled_columns_xtest.transpose()
#
#
# test_mask = ~np.isnan(X_test).any(axis=0)
# X_test = X_test[:, test_mask]
#
# train_mask = ~np.isnan(X_train).any(axis=0)
# X_train = X_train[:, train_mask]
#
#
# callback = keras.callbacks.EarlyStopping(monitor='val_loss',
# 										 # min_delta=0.005,
# 										 patience=20,
# 										 mode='min',
# 										 start_from_epoch=3,
# 										 restore_best_weights=True)
#
#
#
# model =keras.Sequential()
# model.add(keras.layers.Dense(500, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
# model.add(keras.layers.Dense(475, activation='relu'))
# model.add(keras.layers.Dense(375, activation='relu'))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(270, activation='relu'))
# model.add(keras.layers.Dense(140, activation='relu'))
# model.add(keras.layers.Dense(120, activation='relu'))
# model.add(keras.layers.Dense(1, activation='linear'))
# model.compile(loss='MeanSquaredError', optimizer='adam')
#
#
# import datetime
# trainstart = time.time()
# history = model.fit(X_train,
# 					y_train,
# 					epochs=150,
# 					batch_size=32,
# 					callbacks=callback,
# 					validation_data=(X_test, y_test),
# 					verbose=1)
#
# train_end = time.time()
# print(f'Training completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
# predictions = model.predict(X_test)
# predictions = predictions.ravel()
#
#
# rescaled_predictions = []
# predictions_list = predictions.tolist()
#
# for pred in predictions_list:
# 	descaled_p = pred * keff_std + keff_mean
# 	rescaled_predictions.append(float(descaled_p))
#
# errors = []
# for predicted, true in zip(rescaled_predictions, keff_test):
# 	errors.append((predicted - true) * 1e5)
# 	print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')
#
# sorted_errors = sorted(errors)
# absolute_errors = [abs(x) for x in sorted_errors]
# print(f'Average absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')
#
# print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')
#
#
# print(f"Smallest absolute error: {min(absolute_errors)} pcm")
# acceptable_predictions = []
# borderline_predictions = []
# for x in absolute_errors:
# 	if x <= 5.0:
# 		acceptable_predictions.append(x)
# 	if x <= 10.0:
# 		borderline_predictions.append(x)
#
#
# print(f' {len(acceptable_predictions)} ({len(acceptable_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 5 pcm error')
# print(f' {len(borderline_predictions)} ({len(borderline_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 10 pcm error')
#
