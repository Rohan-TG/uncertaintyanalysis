import os
# num_threads = 30
# os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
# os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
# os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"
# os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_threads}"
# os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_threads}"
# import sys
import matplotlib.pyplot as plt


# MLP

# computer = os.uname().nodename
# if computer == 'fermiac':
# 	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
# elif computer == 'oppie':
# 	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore, norm
import tqdm
import keras
import time
import tensorflow as tf


print(tf.config.list_physical_devices('GPU'))


data_directory = input('\n\nData directory: ')

# data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/random/pu-only/all-channels/0-4999/xserg_data'

all_parquets = os.listdir(data_directory)

training_fraction = float(input('\nEnter training data fraction: '))
lower_energy_bound = float(input('\nEnter lower energy bound in eV: '))
patience = float(input('\nEnter patience: '))

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

print('\nFetching training data...')





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

	# xsobject = pu9_mt2xs + pu9_mt4xs +  pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt18xs + pu1_mt102xs
	xsobject = pu9_mt2xs + pu9_mt4xs + pu9_mt16xs + pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt16xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt16xs + pu1_mt18xs + pu1_mt102xs

	XS_obj = xsobject

	return(XS_obj, keff_value)



keff_train = [] # k_eff labels
XS_train = []


for train_file in tqdm.tqdm(training_files, total=n_training_samples):
	xs_values, keff_value = fetch_data(train_file)

	XS_train.append(xs_values)
	keff_train.append(keff_value)

XS_train = np.array(XS_train)
y_train = zscore(keff_train)

train_labels_mean = np.mean(keff_train)
train_labels_std = np.std(keff_train)


scaling_matrix_xtrain = XS_train.transpose()

scaled_columns_xtrain = []
print('\nScaling training data...')


le_bound_index = 1 # filters out NaNs


training_column_means = []
training_column_stds = []

for column in tqdm.tqdm(scaling_matrix_xtrain[le_bound_index:-1], total=len(scaling_matrix_xtrain[le_bound_index:-1])):
	scaled_column = zscore(column)

	column_mean = np.mean(column)
	column_std = np.std(column)
	training_column_means.append(column_mean)
	training_column_stds.append(column_std)
	scaled_columns_xtrain.append(scaled_column)

scaled_columns_xtrain = np.array(scaled_columns_xtrain)
X_train = scaled_columns_xtrain.transpose()


XS_test = []
keff_test = []

print('\nFetching test data...')


for test_file in tqdm.tqdm(test_files, total=len(test_files)):
	xs_values_test, keff_value_test = fetch_data(test_file)
	XS_test.append(xs_values_test)
	keff_test.append(keff_value_test)

XS_test = np.array(XS_test)
# keff_mean = np.mean(keff_test)
# keff_std = np.std(keff_test)
y_test = (np.array(keff_test) - train_labels_mean) / train_labels_std

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
print('\nScaling test data...')
for column, c_mean, c_std in tqdm.tqdm(zip(scaling_matrix_xtest[le_bound_index:-1], training_column_means, training_column_stds), total=len(scaling_matrix_xtest[le_bound_index:-1])):
	# scaled_column = zscore(column)

	scaled_column = (np.array(column) - c_mean) / c_std
	scaled_columns_xtest.append(scaled_column)

scaled_columns_xtest = np.array(scaled_columns_xtest)
X_test = scaled_columns_xtest.transpose()


# test_mask = ~np.isnan(X_test).any(axis=0)
# X_test = X_test[:, test_mask]
X_test = np.nan_to_num(X_test, nan=0.0)

# train_mask = ~np.isnan(X_train).any(axis=0)
# X_train = X_train[:, train_mask]
X_train = np.nan_to_num(X_train, nan=0.0)


callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=patience,
										 mode='min',
										 restore_best_weights=True)





model =keras.Sequential()
model.add(keras.layers.Dense(1000, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(900, activation='relu'))
model.add(keras.layers.Dense(750, activation='relu'))
model.add(keras.layers.Dense(600, activation='relu'))
model.add(keras.layers.Dense(540, activation='relu'))
model.add(keras.layers.Dense(380, activation='relu'))
model.add(keras.layers.Dense(280, activation='relu'))
model.add(keras.layers.Dense(150, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='MeanSquaredError', optimizer='adam')

# model =keras.Sequential()


import datetime
trainstart = time.time()
history = model.fit(X_train,
					y_train,
					epochs=3000,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_test, y_test),
					verbose=1)

train_end = time.time()
print(f'\nTraining completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
predictions = model.predict(X_test)
predictions = predictions.ravel()


rescaled_predictions = []
predictions_list = predictions.tolist()

for pred in predictions_list:
	descaled_p = pred * train_labels_std + train_labels_mean
	rescaled_predictions.append(float(descaled_p))

errors = []
for predicted, true in zip(rescaled_predictions, keff_test):
	errors.append((predicted - true) * 1e5)
	print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

sorted_errors = sorted(errors)
absolute_errors = [abs(x) for x in sorted_errors]
print(f'\nAverage absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')

print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')


print(f"Smallest absolute error: {min(absolute_errors)} pcm")
acceptable_predictions = []
borderline_predictions = []
fifteen_pcm_predictions = []
twenty_pcm_predictions = []
for x in absolute_errors:
	if x <= 5.0:
		acceptable_predictions.append(x)
	if x <= 10.0:
		borderline_predictions.append(x)
	if x <= 15.0 :
		fifteen_pcm_predictions.append(x)
	if x <= 20.0:
		twenty_pcm_predictions.append(x)


print(f' {len(acceptable_predictions)} ({len(acceptable_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 5 pcm error')
print(f' {len(borderline_predictions)} ({len(borderline_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 10 pcm error')
print(f' {len(fifteen_pcm_predictions)} ({len(fifteen_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 15 pcm error)')
print(f' {len(twenty_pcm_predictions)} ({len(twenty_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 20 pcm error)')


plt.figure()
plt.hist(sorted_errors, bins=30)
plt.grid()
plt.title('Distribution of errors')
plt.xlabel('Error / pcm')
plt.ylabel('Count')
# plt.savefig('absolute_errors_corrected_scaling.png', dpi = 300)
plt.show()




skew_positive = []
skew_negative = []

for x in errors:
	if x >0:
		skew_positive.append(x)
	else:
		skew_negative.append(x)

plt.figure()
plt.plot(keff_test, errors, 'x')
plt.grid()
plt.title('Distribution of errors')
plt.xlabel('True k_eff')
plt.ylabel('Error / pcm')
# plt.savefig('errors_as_function_of_keff.png', dpi = 300)
plt.show()

rmse_correction = input('RMSE correction y/n: ')
if rmse_correction == 'y':
	rmse_correction_errors = []
	for predicted, true in zip(rescaled_predictions, keff_test):
		calculated_error = (predicted - true) * 1e5

		if abs(calculated_error) > 5:
			corrected_error = (calculated_error ** 2 - 5 ** 2) ** 0.5
			if calculated_error < 0:
				corrected_error = corrected_error * -1
		else:
			corrected_error = calculated_error

		rmse_correction_errors.append(corrected_error)
		print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {corrected_error:0.0f} pcm')

	absolute_corrected_errors = [abs(x) for x in rmse_correction_errors]
	print(
		f'\nAverage absolute error: {np.mean(absolute_corrected_errors):0.1f} +- {np.std(absolute_corrected_errors):0.1f} pcm')

	acceptable_corrected_predictions = []
	borderline_corrected_predictions = []
	fifteen_corrected_predictions = []
	twenty_corrected_predictions = []
	for x in absolute_corrected_errors:
		if x <= 5.0:
			acceptable_corrected_predictions.append(x)
		if x <= 10.0:
			borderline_corrected_predictions.append(x)
		if x <= 15.0:
			fifteen_corrected_predictions.append(x)
		if x <= 20.0:
			twenty_corrected_predictions.append(x)

	print(f' {len(acceptable_corrected_predictions)} ({len(acceptable_corrected_predictions) / len(absolute_corrected_errors) * 100:.2f}%) predictions <= 5 pcm error')
	print(f' {len(borderline_corrected_predictions)} ({len(borderline_corrected_predictions) / len(absolute_corrected_errors) * 100:.2f}%) predictions <= 10 pcm error')
	print(f' {len(fifteen_corrected_predictions)} ({len(fifteen_corrected_predictions) / len(absolute_corrected_errors) * 100:.2f}%) predictions <= 15 pcm error')
	print(f' {len(twenty_corrected_predictions)} ({len(twenty_corrected_predictions) / len(absolute_corrected_errors) * 100:.2f}%) predictions <= 20 pcm error')














print('Gaussian correction:')
gaussian_correction_probabilities = []
for predicted, true in zip(rescaled_predictions, keff_test):
	calculated_error = (predicted - true) * 1e5

	corrected_gaussian_error = norm.cdf((calculated_error + 10) / 5) - norm.cdf((calculated_error - 10) / 5)

	gaussian_correction_probabilities.append(corrected_gaussian_error)
	print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {calculated_error:0.0f} pcm, Chance: {corrected_gaussian_error* 100:0.1f}%')

expectation_value = np.sum(corrected_gaussian_error)
fractional_expectation = expectation_value / len(keff_test) * 100

