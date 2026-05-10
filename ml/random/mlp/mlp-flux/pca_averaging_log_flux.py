import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
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
import pickle
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA


xs_directory = input('\nXS directory: ')
flux_data_directory = input('Flux data directory: ')

test_xs_directory = input('\nTest XS directory (x for set to val): ')
test_flux_data_directory = input('Test flux data directory (x for set to val): ')
patience = int(input('\nPatience: '))
n_models = int(input('\nN. models: '))
keep_n = int(input('\nKeep n. models: '))


# data_processes = int(input('Num. data processors: '))
data_processes = 6
all_parquets = os.listdir(xs_directory)

training_fraction = float(input('\nEnter training data fraction: '))
lower_energy_bound = float(input('\nEnter lower energy bound in eV: '))

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

average_performance_list = []
average_performance_list_test = []




def fetch_data(datafile, xs_directory, flux_data_directory):

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
	futures = [executor.submit(fetch_data, train_file, xs_directory, flux_data_directory) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, flux_values, flux_error = future.result()
		XS_train.append(xs_values)
		flux_train.append(flux_values)
		flux_train_error.append(flux_error)

XS_train = np.array(XS_train) # shape (num_samples, num_channels, points per channel)
flux_train = np.array(flux_train) # matrix
flux_train_error = np.array(flux_train_error)


# Base flux scraping
base_flux_file = pd.read_parquet('data/Flux_data_Pu-239_-3_Pu-240_-3_Pu-241_-3.parquet', engine='pyarrow')
base_flux = np.array(base_flux_file['flux'].values)
normalised_base_flux = base_flux / (np.sum(base_flux))

def scale_flux(flux_array, flux_error_array, train_mode = False, means = None, stds = None, normalise = True):
	"""setting train_mode to True just makes this function return the means and stds. Otherwise not returned"""

	normalised_flux_array = []
	normalised_flux_error_array = []
	if normalise:
		for flux_set, flux_error_set in zip(flux_array, flux_error_array):
			area = np.sum(flux_set)
			norm_flux_vector = np.array(flux_set) / area
			norm_flux_error = np.array(flux_error_set) / area

			norm_flux_differences = np.log(norm_flux_vector / normalised_base_flux)

			normalised_flux_array.append(norm_flux_differences)
			normalised_flux_error_array.append(norm_flux_error)

		normalised_flux_array = np.array(normalised_flux_array)
		normalised_flux_error_array = np.array(normalised_flux_error_array)

	else:
		flux_differences = np.log(np.array(flux_array) / base_flux)
		# flux_differences = np.array(flux_array) / base_flux
		normalised_flux_array = flux_differences

		normalised_flux_error_array = np.array(flux_error_array)

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

def delogger(logged_array):
	return np.exp(np.array(logged_array))

def descaler(scaled_flux_array, means, stds):
	transposed_flux_array = scaled_flux_array.transpose()
	rescaled_flux_array = []
	for energy_column, mean, std in zip(transposed_flux_array, means, stds):
		rescaled_flux_array.append(energy_column * std + mean)

	rescaled_flux_array = np.array(rescaled_flux_array)
	rescaled_flux_array = rescaled_flux_array.transpose()
	rescaled_flux_array = delogger(rescaled_flux_array)

	return rescaled_flux_array


y_train, scaling_means, scaling_stds, flux_errors_train = scale_flux(flux_train, flux_error_array=flux_train_error, train_mode=True, normalise=True)
pca = PCA(n_components=0.999, svd_solver='full')
Z_train = pca.fit_transform(y_train)


XS_val = []
flux_val = []
flux_val_error = []
print('\nFetching val data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, val_file, xs_directory, flux_data_directory) for val_file in val_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, flux_value_val, flux_error_val = future.result()
		XS_val.append(xs_values_val)
		flux_val.append(flux_value_val)
		flux_val_error.append(flux_error_val)

XS_val = np.array(XS_val)

y_val, flux_errors_val = scale_flux(flux_val, flux_error_array=flux_val_error, train_mode=False, means=scaling_means, stds=scaling_stds, normalise=True)
y_val = np.array(y_val)
Z_val = pca.transform(y_val)

flux_errors_val = np.array(flux_errors_val)

if test_flux_data_directory != 'x':
	print('\nFetching test data...')
	test_files = os.listdir(test_xs_directory)

	XS_test = []
	flux_test = []
	flux_test_error = []
	with ProcessPoolExecutor(max_workers=data_processes) as executor:
		futures_test = [executor.submit(fetch_data, test_file, test_xs_directory, test_flux_data_directory) for test_file in test_files]

		for future_test in tqdm.tqdm(as_completed(futures_test), total=len(futures_test)):
			xs_values_test, flux_value_test, flux_test_err = future_test.result()
			XS_test.append(xs_values_test)
			flux_test.append(flux_value_test)
			flux_test_error.append(flux_test_err)

	XS_test = np.array(XS_test)
	y_test, flux_errors_test = scale_flux(flux_test, flux_error_array=flux_test_error, train_mode=False, means=scaling_means, stds=scaling_stds)


print('\nScaling training data...')


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



if test_flux_data_directory == 'x':
	XS_test = XS_val
	y_test = y_val

X_train, X_val = process_data_mlp(XS_train, XS_val)

def build_model():
	"""Returns a Keras Sequential model"""
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
	model.add(keras.layers.Dense(Z_val.shape[1], activation='linear'))
	model.compile(loss='MeanSquaredError', optimizer='adam')

	return model, callback


model_list = []

prediction_matrix_val = [[] for i in range(len(y_val))]
error_matrix_val = [[] for i in range(len(y_val))]

prediction_matrix_test = [[] for i in range(len(y_test))]
error_matrix_test = [[] for i in range(len(y_test))]

pct_matrix_val = [[] for i in range(len(y_val))]
pct_matrix_test = [[] for i in range(len(y_test))]

over_limit_val_matrix = []

percentage_error_matrix_val = []

true_percentage_uncertainty_matrix_val = [] # matrix containing all the values of the monte carlo uncertainty. used to compare to the averaged predictions post-training loop


# Run training N times with new models and same data
for num in tqdm.tqdm(range(n_models)):
	keras.backend.clear_session()
	temp_model, callback = build_model()

	trainstart = time.time()
	history = temp_model.fit(X_train,
						y_train,
						epochs=3000,
						batch_size=32,
						callbacks=callback,
						validation_data=(X_val, y_val),
						verbose=1)

	train_end = time.time()
	print(f'\nTraining completed in {datetime.timedelta(seconds=(train_end - trainstart))}')

	predictions_val_pca = temp_model.predict(X_val)
	predictions_val = pca.inverse_transform(predictions_val_pca)
	r2s = []


	rescaled_full_p = [] # contains predictions
	rescaled_y_val = [] # contains labels
	pct_list = []
	over_limit_list = []
	for idx, (p_set, true_set) in enumerate(zip(predictions_val, y_val)):
		rescaled_predictions = descaler(p_set, means=scaling_means, stds=scaling_stds)
		rescaled_full_p.append(rescaled_predictions)

		rescaled_true_set = descaler(true_set, means=scaling_means, stds=scaling_stds)
		rescaled_y_val.append(rescaled_true_set)

		# true_percentage_errors = 100 * (np.array(flux_errors_val[idx]) / np.array(rescaled_true_set))
		norm_flux_val = flux_val[idx] / np.sum(flux_val[idx])
		true_pct_uncertainty = 100 * (np.array(flux_errors_val[idx]) / np.array(norm_flux_val))
		true_percentage_uncertainty_matrix_val.append(true_pct_uncertainty)

		ratios = np.array(rescaled_predictions) / np.array(rescaled_true_set)
		pct_deviation = (ratios - 1.0) * 100
		pct_list.append(pct_deviation)

		count = 0
		for point_ml_error, point_real_error in zip(pct_deviation, true_pct_uncertainty):
			if abs(point_ml_error) > abs(point_real_error * 2):
				count += 1

		over_limit_pct = (count / len(pct_deviation)) * 100
		over_limit_list.append(over_limit_pct)

		over_limit_val_matrix.append(over_limit_list)
		# print(f'{idx} - {over_limit_pct:0.1f}% Points over limit, Mean: {np.mean(ratios):0.4f} Max: {max(ratios):0.4f} Min: {min(ratios):0.4f} R2: {r2_score(rescaled_predictions, rescaled_true_set):0.5f}')

		prediction_matrix_val[idx].append(rescaled_predictions)

		pct_matrix_val[idx].append(pct_deviation) # working with this

	r2s.append(r2_score(rescaled_predictions, rescaled_true_set))

# pct_matrix_val has shape (n_samples, n_models, 300)
print('\nEvaluating averaged predictions...')
averaged_predictions_val = []

for post_idx, (sample, mc_uncertainties) in enumerate(zip(pct_matrix_val, true_percentage_uncertainty_matrix_val)):
	transposed_sample = np.array(sample).transpose()
	averaged_sample_prediction = []
	for energy_point_vector in transposed_sample:
		mean_pct_deviation = np.mean(energy_point_vector)
		averaged_sample_prediction.append(mean_pct_deviation)

	averaged_predictions_val.append(averaged_sample_prediction)

	avg_count = 0
	for pred_point, true_uncert in zip(averaged_sample_prediction, mc_uncertainties):
		if abs(pred_point) > abs(true_uncert * 2):
			avg_count += 1

	avg_limit_pct = (avg_count / len(mc_uncertainties)) * 100
	print(f'{post_idx} - {avg_limit_pct:0.1f}% Points over limit')


print(f'\n\nMean R2: {np.mean(r2s):0.5f}')
print(f'Avg. {np.mean(over_limit_list):0.1f}% +- {np.std(over_limit_list):0.1f}% over limit')





def count_outliers(prediction_list, labels, flux_errors_list):
	"""prediction_list: normal scale predictions
	labels: unscaled (e.g. y_val)"""
	pct_list = []
	exceeded_limit_list = []
	for idx, (p_set, true_set) in enumerate(zip(prediction_list, labels)):

		rescaled_labels = descaler(true_set, means=scaling_means, stds=scaling_stds)

		label_pct_errors = 100 * (np.array(flux_errors_list[idx]) / np.array(p_set))

		ratios = np.array(p_set) / np.array(rescaled_labels)

		pct_deviation = (ratios - 1.0) * 100
		pct_list.append(pct_deviation)

		count = 0
		for point_ml_err, point_real_err in zip(pct_deviation, label_pct_errors):
			if abs(point_ml_err) > abs(point_real_err * 2):
				count += 1

		exceeded_limit_pct = (count / len(pct_deviation)) * 100
		exceeded_limit_list.append(exceeded_limit_pct)


# flux_file = 'Flux_data_Pu-239_9173_Pu-240_9113_Pu-241_9675.parquet'
# flux_read_obj = pd.read_parquet(f'{flux_data_directory}/{flux_file}', engine='pyarrow')
# flux_lower_bounds = flux_read_obj['low_erg_bounds'].values
# flux_upper_bounds = flux_read_obj['high_erg_bounds'].values

# d1, d2, d3 = fetch_data(all_parquets[0])
#
# grid = []
# for i in flux_lower_bounds:
# 	grid.append(float(i))
#
# for j in flux_upper_bounds:
# 	grid.append(float(j))
#
# grid.sort()
# edges = []
# for i in grid:
# 	if i not in edges:
# 		edges.append(i)
# widths = np.diff(edges)



# def plot_index(sample_idx):
#
# 	true_pct_error = 100 * (np.array(flux_errors_val[sample_idx]) / np.array(rescaled_y_val[sample_idx]))
#
# 	sigma_2_upper = 2* true_pct_error
# 	sigma_2_lower = -2 * true_pct_error
#
#
#
# 	plt.figure()
# 	plt.bar(edges[:-1], rescaled_full_p[sample_idx], width=widths, label='Prediction')
# 	plt.bar(edges[:-1], rescaled_y_val[sample_idx], width=widths, label = "True")
# 	plt.grid()
# 	plt.legend()
# 	plt.xlabel('Energy / MeV')
# 	plt.ylabel('Normalised flux')
# 	plt.xscale('log')
# 	plt.ylim(0,0.015)
# 	plt.savefig(f'{sample_idx}_bar_mlp.png')
#
#
# 	scale_log = input('Log scale? (y): ')
# 	plt.figure()
# 	plt.bar(edges[:-1], pct_list[sample_idx], width=widths, label = 'ML error')
# 	plt.plot(edges[:-1], true_pct_error, label='MC Error')
# 	plt.fill_between(edges[:-1], sigma_2_lower, sigma_2_upper, color='r', alpha=0.3, label = '2$\sigma$')
# 	plt.xlabel('Energy / Mev')
# 	plt.ylabel('% Deviation')
# 	if scale_log == 'y':
# 		plt.xscale('log')
# 	plt.grid()
# 	plt.legend()
# 	plt.savefig(f'{sample_idx}_val_pct_error_bar_mlp.png')