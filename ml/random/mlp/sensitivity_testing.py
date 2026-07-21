import os
import matplotlib.pyplot as plt
import keras.backend
import pickle
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore, norm
import tqdm
import keras
import time
import tensorflow as tf
import datetime


model_directory = input('\nModel directory: ')
with open(f"{model_directory}/train_labels_std.pkl", "rb") as f:
	train_labels_std = pickle.load(f)

with open(f"{model_directory}/train_labels_mean.pkl", "rb") as f:
	train_labels_mean = pickle.load(f)

with open(f"{model_directory}/training_columns_means.pkl", "rb") as f:
	training_column_means = pickle.load(f)

with open(f"{model_directory}/training_columns_stds.pkl", "rb") as f:
	training_column_stds = pickle.load(f)


contents = os.listdir(model_directory)
models = []
for object in contents:
	if object.endswith(".keras"):
		models.append(keras.models.load_model(f"{model_directory}/{object}"))

data_directory = input('\n\nXS data directory: ')
test_directory = input('\nTest directory (x set to val): ')
keep_n = int(input('\nN models: '))
if test_directory != 'x':
	test_files = os.listdir(test_directory)
le_bound_index = 1 # filters out NaNs


all_parquets = os.listdir(data_directory)
lower_energy_bound = float(input('\nEnter lower energy bound in eV: '))


list_of_channels = ['94239_MT2_XS', '94239_MT4_XS', '94239_MT16_XS', '94239_MT18_XS', '94239_MT102_XS',
					'94240_MT2_XS', '94240_MT4_XS', '94240_MT16_XS', '94240_MT18_XS', '94240_MT102_XS',
					'94241_MT2_XS', '94241_MT4_XS', '94241_MT16_XS', '94241_MT18_XS', '94241_MT102_XS',]

isolated_channels = []


def fetch_data(datafile, dir=data_directory):
	baseline_df = pd.read_parquet(
		'/home/rnt26/uncertaintyanalysis/ml/mldata/baselines/endfbviii.0/endfbviii0_baseline_data_Pu-239_-1_Pu-240_-1_Pu-241_-1.parquet',
		engine='pyarrow')

	isolated_df = pd.read_parquet(f'{data_directory}/{datafile}', engine='pyarrow')

	baseline_df = baseline_df[baseline_df['ERG'] >= lower_energy_bound] # Set energy domain
	isolated_df = isolated_df[isolated_df['ERG'] >= lower_energy_bound]

	keff_value = float(isolated_df['keff'].values[0])

	unflattened_matrix = [[] for _ in list_of_channels] # initialise data matrix

	for channel_index, channel in enumerate(list_of_channels):
		if channel not in isolated_channels: # check if data not being isolated
			unflattened_matrix[channel_index] = baseline_df[channel].values.tolist()
		else: # if data isolated then do this
			unflattened_matrix[channel_index] = isolated_df[channel].values.tolist()

	XS_obj = np.array(unflattened_matrix).flatten() # turn into vector

	return XS_obj, keff_value



# def fetch_data(datafile, dir = data_directory):
#
# 	temp_df = pd.read_parquet('/home/rnt26/uncertaintyanalysis/ml/mldata/baselines/endfbviii.0/endfbviii0_baseline_data_Pu-239_-1_Pu-240_-1_Pu-241_-1.parquet', engine='pyarrow')
# 	testing_df = pd.read_parquet(f'{data_directory}/{datafile}', engine='pyarrow')
#
# 	testing_df = testing_df[testing_df['ERG'] >= lower_energy_bound]
# 	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]
#
# 	keff_value = float(temp_df['keff'].values[0])
#
# 	pu9_mt18xs = temp_df['94239_MT18_XS'].values.tolist()
# 	pu0_mt18xs = temp_df['94240_MT18_XS'].values.tolist()
# 	pu1_mt18xs = temp_df['94241_MT18_XS'].values.tolist()
#
# 	pu9_mt2xs = testing_df['94239_MT2_XS'].values.tolist()
# 	pu0_mt2xs = temp_df['94240_MT2_XS'].values.tolist()
# 	pu1_mt2xs = temp_df['94241_MT2_XS'].values.tolist()
#
# 	pu9_mt4xs = temp_df['94239_MT4_XS'].values.tolist()
# 	pu0_mt4xs = temp_df['94240_MT4_XS'].values.tolist()
# 	pu1_mt4xs = temp_df['94241_MT4_XS'].values.tolist()
#
# 	pu9_mt16xs = temp_df['94239_MT16_XS'].values.tolist()
# 	pu0_mt16xs = temp_df['94240_MT16_XS'].values.tolist()
# 	pu1_mt16xs = temp_df['94241_MT16_XS'].values.tolist()
#
# 	pu9_mt102xs = temp_df['94239_MT102_XS'].values.tolist()
# 	pu0_mt102xs = temp_df['94240_MT102_XS'].values.tolist()
# 	pu1_mt102xs = temp_df['94241_MT102_XS'].values.tolist()
#
# 	# xsobject = pu9_mt2xs + pu9_mt4xs +  pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt18xs + pu1_mt102xs
# 	xsobject = pu9_mt2xs + pu9_mt4xs + pu9_mt16xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt16xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt16xs + pu1_mt18xs + pu1_mt102xs
#
# 	XS_obj = xsobject
#
# 	return(XS_obj, keff_value)

average_performance_list_test = []

keff_test = []
XS_test = []
pu9_test_indices = []
pu0_test_indices = []
pu1_test_indices = []
for test_file in tqdm.tqdm(test_files, total=len(test_files)):
	xs_values, keff_value = fetch_data(test_file, dir=test_directory)

	XS_test.append(xs_values)
	keff_test.append(keff_value)

	# pu9_test_index = int(test_file.split('_')[4])
	# pu9_test_indices.append(pu9_test_index)

	pu0_test_index = int(test_file.split('_')[6])
	pu0_test_indices.append(pu0_test_index)

	# pu1_test_index = int(test_file.split('_')[8].split('.')[0])
	# pu1_test_indices.append(pu1_test_index)


XS_test = np.array(XS_test)
y_test = (np.array(keff_test) - train_labels_mean) / train_labels_std

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
print('\nScaling test data...')
for column, c_mean, c_std in tqdm.tqdm(zip(scaling_matrix_xtest[le_bound_index:-1], training_column_means, training_column_stds), total=len(scaling_matrix_xtest[le_bound_index:-1])):
	scaled_column = (np.array(column) - c_mean) / c_std
	scaled_columns_xtest.append(scaled_column)

scaled_columns_xtest = np.array(scaled_columns_xtest)
X_test = scaled_columns_xtest.transpose()
X_test = np.nan_to_num(X_test, nan=0.0)

prediction_matrix_test = [[] for i in range(len(y_test))]
error_matrix_test = [[] for i in range(len(y_test))]

for m in models:
	predictions_test = m.predict(X_test)
	predictions_test = predictions_test.ravel()
	rescaled_predictions_test = []
	predictions_list_test = predictions_test.tolist()
	for p in predictions_list_test:
		descaled_p = p * train_labels_std + train_labels_mean
		rescaled_predictions_test.append(float(descaled_p))

	errors_test = []
	for predicted, true in zip(rescaled_predictions_test, keff_test):
		errors_test.append((predicted - true) * 1e5)

	for p_i, p in enumerate(rescaled_predictions_test):
		prediction_matrix_test[p_i].append(p)

	for err_index, err in enumerate(errors_test):
		error_matrix_test[err_index].append(err)

	absolute_errors_test = [abs(x) for x in errors_test]
	average_performance_list_test.append(np.mean(absolute_errors_test))




def select_best_models(error_matrix, keep_n_models, threshold=10, mode='test'):
	"""error_matrix: the error matrix
	keep_n_models: the number of models to keep"""

	truncated_count_threshold = 0
	em = np.array(error_matrix)
	em_trans = np.transpose(em)

	model_averages = []
	for model_ in em_trans:
		model_averages.append(np.mean(np.abs(model_)))

	sorted_models = [[-1,100]]
	for i, val in enumerate(model_averages):
		for j, saved in enumerate(sorted_models):
			if val < saved[-1]:
				sorted_models.insert(j, [i,val])
				break

	acceptable_models = []
	for x in sorted_models[:keep_n_models]:
		acceptable_models.append(x[0])

	if mode == 'test':
		emt = np.array(error_matrix_test)
	# else:
		# emt = np.array(error_matrix_val)

	best_averaged_errors = []

	for sample in emt:
		working_list = []
		for model_index, value in enumerate(sample):
			if model_index in acceptable_models:
				working_list.append(value)

		if np.abs(np.mean(working_list)) <= threshold:
			truncated_count_threshold +=1

		best_averaged_errors.append(np.mean(working_list))


	return acceptable_models, truncated_count_threshold, best_averaged_errors



best_models, best_models_count10, averaged_errors = select_best_models(error_matrix_test, keep_n)
print(best_models_count10 / len(keff_test) * 100)



def k11():
	"""2nd order term calculation for a1"""
	pass

def k22():
	"""2nd order term calculation for a2"""
	pass

def k12():
	"""2nd order joint term calculation for a1 and a2"""
	pass
