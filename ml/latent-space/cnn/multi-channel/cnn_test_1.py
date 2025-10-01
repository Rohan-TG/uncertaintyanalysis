import pandas as pd
import keras
import numpy as np
import os
import time
from scipy.stats import zscore
import random
import tqdm
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')
from groupEnergies import Groups
data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/g4_fission_elastic_pu9'


g4boundary = Groups.g4
g3boundary = Groups.g3

all_parquets = os.listdir(data_directory)

training_fraction = 0.95
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


keff_train = [] # k_eff labels

XS_train = []

for file in tqdm.tqdm(training_files, total=len(training_files)):
	dftrain = pd.read_parquet(f'{data_directory}/{file}', engine='pyarrow') # Fetch data from parquet file

	keff_train += [float(dftrain['keff'].values[0])] # append k_eff value from the file

	dftrain = dftrain[dftrain.ERG >= g4boundary]
	dftrain = dftrain[dftrain.ERG <= g3boundary]

	mt18xs = np.log(dftrain['MT18_XS'].values) # appends a list of fission cross sections to the XS_fission_train matrix

	mt2xs = np.log(dftrain['MT2_XS'].values) # likewise for elastic scattering cross sections

	mt18xs = mt18xs.tolist()
	mt2xs = mt2xs.tolist()

	XS_train.append([mt18xs, mt2xs])

XS_train = np.array(XS_train)
y_train = np.array(keff_train)
y_train = zscore(y_train)

keff_train_mean = np.mean(keff_train)
keff_train_std = np.std(keff_train)
#
# scaling_matrix_xtrain = XS_train.transpose()
#
# scaled_columns_xtrain = []
# for sample in tqdm.tqdm(scaling_matrix_xtrain, total=len(scaling_matrix_xtrain)):
# 	for feature_column in sample:
# 		scaled_column = zscore(column)
# 		scaled_columns_xtrain.append(scaled_column)
#
# scaled_columns_xtrain = np.array(scaled_columns_xtrain)
# Transposed_scaled_xtrain = scaled_columns_xtrain.transpose()
#
# X_train = []
# for i in Transposed_scaled_xtrain:
# 	row = i[~np.isnan(i)]
# 	X_train.append(row)
#
# X_train = np.array(X_train)
#
#
# print('Training data processed...')
#
#
# ########################################## Test data preparation #######################################################
# XS_test = []
# keff_test = []
# for testfile in tqdm.tqdm(test_files, total=len(test_files)):
# 	dftest = pd.read_parquet(f'{data_directory}/{testfile}', engine='pyarrow')
# 	keff_test += [float(dftest['keff'].values[0])]
#
# 	dftest = dftest[dftest.ERG >= g4boundary]
# 	dftest = dftest[dftest.ERG <= g3boundary]
#
# 	mt18xstest = np.log(dftest['MT18_XS'].values)  # appends a list of fission cross sections to the XS_fission_train matrix
#
# 	mt2xstest = np.log(dftest['MT2_XS'].values)  # likewise for elastic scattering cross sections
#
# 	mt18xstest = mt18xstest.tolist()
# 	mt2xstest = mt2xstest.tolist()
#
# 	XS_test.append(xsobjecttest)
#
# XS_test = np.array(XS_test)
# keff_mean = np.mean(keff_test)
# keff_std = np.std(keff_test)
# y_test = zscore(keff_test)
#
# scaling_matrix_xtest = XS_test.transpose()
#
# scaled_columns_xtest = []
# for columntest in tqdm.tqdm(scaling_matrix_xtest[1:], total=len(scaling_matrix_xtest[1:])):
# 	scaled_column_test = zscore(columntest)
# 	scaled_columns_xtest.append(scaled_column_test)
#
# scaled_columns_xtest = np.array(scaled_columns_xtest)
# Transposed_scaled_xtest = scaled_columns_xtest.transpose()
#
# X_test = []
# for i in Transposed_scaled_xtest:
# 	row = i[~np.isnan(i)]
# 	X_test.append(row)
#
# X_test = np.array(X_test)
#
# callback = keras.callbacks.EarlyStopping(monitor='val_loss',
# 										 # min_delta=0.005,
# 										 patience=20,
# 										 mode='min',
# 										 start_from_epoch=5,
# 										 restore_best_weights=True)
#
# model =keras.Sequential()
# model.add(keras.layers.Dense(10, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
# model.add(keras.layers.Dense(10, activation='relu'))
# # model.add(keras.layers.Dense(100, activation='relu'))
# # model.add(keras.layers.Dense(1000, activation='relu'))
# # model.add(keras.layers.Dense(500, activation='relu'))
# model.add(keras.layers.Dense(1, activation='linear'))
# model.compile(loss='MeanSquaredError', optimizer='adam')
#
#
# trainstart = time.time()
# history = model.fit(X_train,
# 					y_train,
# 					epochs=100,
# 					batch_size=4,
# 					callbacks=callback,
# 					validation_data=(X_test, y_test),
# 					verbose=1)
#
#
# train_end = time.time()
# print(f'Training completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
# predictions = model.predict(X_test)
# predictions = predictions.ravel()
#
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
#
# sorted_errors = sorted(errors)
# absolute_errors = [abs(x) for x in sorted_errors]
# print(f'Average absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')
#
# print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')
#
#
# print(f"Smallest absolute error: {min(absolute_errors)} pcm")