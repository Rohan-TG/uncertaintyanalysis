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
test_data_directory = input('Test data directory (x for set to val): ')

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

val_files = []
for file in all_parquets:
	if file not in training_files:
		val_files.append(file)


print('Fetching training data...')

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
print('Fetching val data...')

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, val_file) for val_file in val_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, keff_value_val = future.result()
		XS_val.append(xs_values_val)
		keff_val.append(keff_value_val)

XS_val = np.array(XS_val)
y_val = (np.array(keff_val) - keff_train_mean) / keff_train_std




###### !!! WIP !!! still not done probably don't use this for a while
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
	print('Fetching test data...')
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


print('Scaling all data...')