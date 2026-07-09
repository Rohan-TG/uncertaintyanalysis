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
if test_directory != 'x':
	test_files = os.listdir(test_directory)
le_bound_index = 1 # filters out NaNs


all_parquets = os.listdir(data_directory)
lower_energy_bound = float(input('\nEnter lower energy bound in eV: '))


def fetch_data(datafile, dir = data_directory):

	temp_df = pd.read_parquet('/home/rnt26/uncertaintyanalysis/ml/mldata/baselines/endfbviii.0/endfbviii0_baseline_data_Pu-239_-1_Pu-240_-1_Pu-241_-1.parquet', engine='pyarrow')
	testing_df = pd.read_parquet(f'{data_directory}/{datafile}', engine='pyarrow')

	testing_df = testing_df[testing_df['ERG'] >= lower_energy_bound]
	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]

	keff_value = float(temp_df['keff'].values[0])

	pu9_mt18xs = temp_df['94239_MT18_XS'].values.tolist()
	pu0_mt18xs = temp_df['94240_MT18_XS'].values.tolist()
	pu1_mt18xs = temp_df['94241_MT18_XS'].values.tolist()

	pu9_mt2xs = testing_df['94239_MT2_XS'].values.tolist()
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



keff_test = []
XS_test = []
pu9_test_indices = []
pu0_test_indices = []
pu1_test_indices = []
for test_file in tqdm.tqdm(test_files, total=len(test_files)):
	xs_values, keff_value = fetch_data(test_file, dir=test_directory)

	XS_test.append(xs_values)
	keff_test.append(keff_value)

	pu9_test_index = int(test_file.split('_')[4])
	pu9_test_indices.append(pu9_test_index)

	pu0_test_index = int(test_file.split('_')[6])
	pu0_test_indices.append(pu0_test_index)

	pu1_test_index = int(test_file.split('_')[8].split('.')[0])
	pu1_test_indices.append(pu1_test_index)


XS_test = np.array(XS_test)
y_test = (np.array(keff_test) - train_labels_mean) / train_labels_std

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
print('\nScaling test data...')
for column, c_mean, c_std in tqdm.tqdm(zip(scaling_matrix_xtest[le_bound_index:-1], training_column_means, training_column_stds), total=len(scaling_matrix_xtest[le_bound_index:-1])):
	scaled_column = (np.array(column) - c_mean) / c_std
	scaled_columns_xtest.append(scaled_column)

# scaled_columns_xtest = np.array(scaled_columns_xtest)
# X_test = scaled_columns_xtest.transpose()