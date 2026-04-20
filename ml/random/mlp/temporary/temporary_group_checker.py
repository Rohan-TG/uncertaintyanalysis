import os
# num_threads = 30
# os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
# os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
# os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"
# os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_threads}"
# os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_threads}"
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
# import shap

# MLP

computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie' or computer == 'bethe':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore
import tqdm
import keras
import time




data_directory = input('\n\nData directory: ')

test_directory = '/home/rnt26/uncertaintyanalysis/ml/mldata/baselines/endfbviii.0'
test_files = os.listdir(test_directory)

data_processes = 6
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

val_files = []
for file in all_parquets:
	if file not in training_files:
		val_files.append(file)



print('\nFetching training data...')
groupdir = '/home/rnt26/uncertaintyanalysis/ml/random/mlp/temporary/groupfiles'
groupfiles = os.listdir(data_directory)

groupfile = f'{groupdir}/Pu-239_g3_-0.011_MT18.parquet'

def fetch_data(datafile, data_dir=data_directory):

	temp_df = pd.read_parquet(f'{data_dir}/{datafile}', engine='pyarrow')
	energies = temp_df['ERG'].values
	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]

	keff_value = float(temp_df['keff'].values[0])


	pu9_mt18xs = temp_df['94239_MT18_XS'].values.tolist()

	if data_dir == test_directory:
		df = pd.read_parquet(groupfile, engine='pyarrow')
		group_reduction = np.interp(energies, df['ERG'].values, df['XS'].values)

		pu9_mt18xs = group_reduction
		keff_value = df['keff'].values[0]

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

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, train_file) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, keff_value = future.result()
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
y_val = (np.array(keff_val) - train_labels_mean) / train_labels_std

scaling_matrix_xval = XS_val.transpose()

scaled_columns_xval = []
print('\nScaling val data...')
for column, c_mean, c_std in tqdm.tqdm(zip(scaling_matrix_xval[le_bound_index:-1], training_column_means, training_column_stds), total=len(scaling_matrix_xval[le_bound_index:-1])):
	# scaled_column = zscore(column)

	scaled_column = (np.array(column) - c_mean) / c_std
	scaled_columns_xval.append(scaled_column)

scaled_columns_xval = np.array(scaled_columns_xval)
X_val = scaled_columns_xval.transpose()

print("\nFetching test data")
XS_test = []
keff_test = []

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, test_file, test_directory) for test_file in test_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_test, keff_value_test = future.result()
		XS_test.append(xs_values_test)
		keff_test.append(keff_value_test)

XS_test = np.array(XS_test)
y_test = (np.array(keff_test) - train_labels_mean) / train_labels_std

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
print('\nScaling test data...')
for column, c_mean, c_std in tqdm.tqdm(
		zip(scaling_matrix_xtest[le_bound_index:-1], training_column_means, training_column_stds),
		total=len(scaling_matrix_xtest[le_bound_index:-1])):

	scaled_column = (np.array(column) - c_mean) / c_std
	scaled_columns_xtest.append(scaled_column)

scaled_columns_xtest = np.array(scaled_columns_xtest)
X_test = scaled_columns_xtest.transpose()

X_test = np.nan_to_num(X_test, nan=0.0)

X_val = np.nan_to_num(X_val, nan=0.0)
X_train = np.nan_to_num(X_train, nan=0.0)


callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=patience,
										 mode='min',
										 start_from_epoch=3,
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




import datetime
trainstart = time.time()
history = model.fit(X_train,
					y_train,
					epochs=3000,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_val, y_val),
					verbose=1)

train_end = time.time()

predictions = model.predict(X_val)
predictions = predictions.ravel()


rescaled_predictions = []
predictions_list = predictions.tolist()

for pred in predictions_list:
	descaled_p = pred * train_labels_std + train_labels_mean
	rescaled_predictions.append(float(descaled_p))

errors = []
for predicted, true in zip(rescaled_predictions, keff_val):
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

print(f'\nTraining completed in {datetime.timedelta(seconds=(train_end - trainstart))}')

save_histogram = input('Save histogram? (y): ')
if save_histogram == 'y':
	plt.figure()
	plt.hist(sorted_errors, bins=30)
	plt.grid()
	plt.title('Distribution of errors')
	plt.xlabel('Error / pcm')
	plt.ylabel('Count')
	plt.savefig('absolute_errors_corrected_scaling.png', dpi = 300)
	plt.show()




skew_positive = []
skew_negative = []

for x in errors:
	if x >0:
		skew_positive.append(x)
	else:
		skew_negative.append(x)

plt.figure()
plt.plot(keff_val, errors, 'x')
plt.grid()
plt.title('Distribution of errors')
plt.xlabel('True k_eff')
plt.ylabel('Error / pcm')
plt.savefig('errors_as_function_of_keff.png', dpi = 300)
plt.show()

### Feature importance

# shap_values = shap.DeepExplainer(model=model, data=X_test)


# if mask != 'x':
# 	masking_value = 0
# 	X_test[np.abs(X_test) >= float(mask)] = masking_value

test_predictions = model.predict(X_test)
test_predictions = test_predictions.ravel()

rescaled_test_predictions = []
test_predictions_list = test_predictions.tolist()

for pred in test_predictions_list:
	descaled_p = pred * train_labels_std + train_labels_mean
	rescaled_test_predictions.append(float(descaled_p))

test_errors = []
for predicted, true in zip(rescaled_test_predictions, keff_test):
	test_errors.append((predicted - true) * 1e5)
	print(f'TEST - SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

sorted_test_errors = sorted(test_errors)
absolute_test_errors = [abs(x) for x in sorted_test_errors]
print(f'\nAverage absolute test error: {np.mean(absolute_test_errors)} +- {np.std(absolute_test_errors)}')

print(f'Max -ve error: {sorted_test_errors[0]} pcm, Max +ve error: {sorted_test_errors[-1]} pcm')

print(f"Smallest absolute error: {min(absolute_test_errors)} pcm")
acceptable_test_predictions = []
borderline_test_predictions = []
fifteen_pcm_test_predictions = []
twenty_pcm_test_predictions = []
for x in absolute_test_errors:
	if x <= 5.0:
		acceptable_test_predictions.append(x)
	if x <= 10.0:
		borderline_test_predictions.append(x)
	if x <= 15.0:
		fifteen_pcm_test_predictions.append(x)
	if x <= 20.0:
		twenty_pcm_test_predictions.append(x)

print(f' {len(acceptable_test_predictions)} ({len(acceptable_test_predictions) / len(absolute_test_errors) * 100:.2f}%) predictions <= 5 pcm error')
print(f' {len(borderline_test_predictions)} ({len(borderline_test_predictions) / len(absolute_test_errors) * 100:.2f}%) predictions <= 10 pcm error')
print(f' {len(fifteen_pcm_test_predictions)} ({len(fifteen_pcm_test_predictions) / len(absolute_test_errors) * 100:.2f}%) predictions <= 15 pcm error)')
print(f' {len(twenty_pcm_test_predictions)} ({len(twenty_pcm_test_predictions) / len(absolute_test_errors) * 100:.2f}%) predictions <= 20 pcm error)')

print('#### End test ####')

### register errors

# dump_directory = input('Dump directory: ')
# RUNCODE = int(input('Run code: '))
# for error, prediction, file in tqdm.tqdm(zip(errors, predictions, val_files), total=len(errors)):
#
#
# 	data_df = pd.read_parquet(f'{data_directory}/{file}', engine='pyarrow')
# 	df2 = data_df.copy()
# 	iterator = list(range(0, len(df2)))
#
# 	pu9_index = int(file.split('_')[4])
# 	pu0_index = int(file.split('_')[6])
# 	pu1_index = int(file.split('_')[8].split('.')[0])
#
# 	index_list_pu9 = [pu9_index for i in iterator]
# 	index_list_pu0 = [pu0_index for i in iterator]
# 	index_list_pu1 = [pu1_index for i in iterator]
#
# 	error_data_list = [error for i in iterator]
# 	prediction_list = [prediction for i in iterator]
#
# 	df2['prediction'] = prediction_list
# 	df2['ml_error'] = error_data_list
# 	df2['pu239_index'] = index_list_pu9
# 	df2['pu240_index'] = index_list_pu0
# 	df2['pu241_index'] = index_list_pu1
#
# 	df2.to_parquet(f'{dump_directory}/diagnosis_data_Pu-239_{pu9_index}_Pu-240_{pu0_index}_Pu-241_{pu1_index}_runcode-{RUNCODE}.parquet',
# 				   engine='pyarrow')
