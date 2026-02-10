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

keff_train_mean = np.mean(keff_train)
keff_train_std = np.std(keff_train)
y_train = zscore(keff_train)

XS_val = []
keff_val = []

print('Fetching test data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, test_file) for test_file in test_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, keff_value_val = future.result()
		XS_val.append(xs_values_val)
		keff_val.append(keff_value_val)

XS_val = np.array(XS_val)
y_val = (np.array(keff_val) - keff_train_mean) / keff_train_std


print('Scaling training data...')


# le_bound_index = 1 # filters out NaNs

def process_data(XS_train, XS_val):


	channel_matrix_train = [[] for i in range(len(XS_train[0]))] # each element is a matrix of only one channel, e.g. channel_matrix[0] is all the lists containing
	channel_matrix_val = [[] for i in range(len(XS_val[0]))]
	# Pu-239 (n,el)
	scaled_channel_matrix_train = []
	scaled_channel_matrix_val = []

	for matrix in tqdm.tqdm(XS_train, total =len(XS_train)):
		# Each matrix has shape (num channels, points per channel)
		for channel_index, channel in enumerate(matrix):
			channel_matrix_train[channel_index].append(channel)

		# channel_matrix now has shape (num channels, num samples, points per channel)
		# Each element of channel matrix has shape (num samples, points per channel)
	for matrix in tqdm.tqdm(XS_val, total =len(XS_val)):
		for channel_index, channel in enumerate(matrix):
			channel_matrix_val[channel_index].append(channel)

	#################################################################################################################################################################
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

	###################################################################################################################################################################
	# print('Forming scaled training data...')
	X_matrix_train = [[] for i in range(XS_train.shape[0])] # number of samples
	X_matrix_val = [[] for i in range(XS_val.shape[0])]

	for scaled_observable in scaled_channel_matrix_train:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_train[sample_index].append(channel_sample)

	for scaled_observable in scaled_channel_matrix_val:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_val[sample_index].append(channel_sample)

	X_matrix_train = np.array(X_matrix_train)
	X_matrix_train[np.isnan(X_matrix_train)] = 0

	X_matrix_val = np.array(X_matrix_val)
	X_matrix_val[np.isnan(X_matrix_val)] = 0 # changes nans to 0

	return X_matrix_train, X_matrix_val

X_train, X_val = process_data(XS_train, XS_val)






callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=50,
										 mode='min',
										 start_from_epoch=3,
										 restore_best_weights=True)



# model = keras.Sequential()
# model.add(keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
# model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',))
# # model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',))
# # model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(500, activation='relu'))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(150, activation='relu'))
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(1, activation='linear'))
# model.compile(loss='MeanSquaredError', optimizer='adam')

model = keras.Sequential()
model.add(keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',))
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
					batch_size=32,
					callbacks=callback,
					validation_data=(X_val, y_val),
					verbose=1)

train_end = time.time()
print(f'Training completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
predictions = model.predict(X_val)
predictions = predictions.ravel()


rescaled_predictions = []
predictions_list = predictions.tolist()

for pred in predictions_list:
	descaled_p = pred * keff_train_std + keff_train_mean
	rescaled_predictions.append(float(descaled_p))

errors = []
for predicted, true in zip(rescaled_predictions, keff_val):
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
	plt.figure()
	plt.hist(sorted_errors, bins=25)
	plt.grid()
	plt.title('Distribution of errors')
	plt.xlabel('Error / pcm')
	plt.ylabel('Count')
	plt.savefig('asdfcnn.png', dpi=300)
	plt.show()

	plt.figure()
	plt.plot(keff_val, errors, 'x')
	plt.grid()
	plt.title('Distribution of errors')
	plt.xlabel('True k_eff')
	plt.ylabel('Error / pcm')
	plt.savefig('errors_as_function_of_keff.png', dpi=300)
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
