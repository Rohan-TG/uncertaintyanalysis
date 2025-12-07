import os
import sys

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




data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/Pu239/random/fission/xserg_data'

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
keff_train = [] # k_eff labels
XS_train = []
for file in tqdm.tqdm(training_files, total=len(training_files)):
	# group = file.split('_')[1][1]

	dftrain = pd.read_parquet(f'{data_directory}/{file}', engine='pyarrow')
	dftrain = dftrain[dftrain['ERG'] >= lower_energy_bound]

	keff_train += [float(dftrain['keff'].values[0])]  # append k_eff value from the file

	mt18xs = dftrain['XS'].values  # appends a list of fission cross sections to the XS_fission_train matrix
	# mt18xs = mt18xs[1:391]

	XS_train.append(mt18xs)

XS_train = np.array(XS_train)
y_train = zscore(keff_train)

scaling_matrix_xtrain = XS_train.transpose()

scaled_columns_xtrain = []
print('Scaling training data...')


le_bound_index = 1 # filters out NaNs




for column in tqdm.tqdm(scaling_matrix_xtrain[le_bound_index:-1], total=len(scaling_matrix_xtrain[le_bound_index:-1])):
	scaled_column = zscore(column)
	scaled_columns_xtrain.append(scaled_column)

scaled_columns_xtrain = np.array(scaled_columns_xtrain)
X_train = scaled_columns_xtrain.transpose()


XS_test = []
keff_test = []

print('Fetching test data...')
for file in tqdm.tqdm(test_files, total=len(test_files)):
	# group = file.split('_')[1][1]
	dftest = pd.read_parquet(f'{data_directory}/{file}', engine='pyarrow')
	dftest = dftest[dftest['ERG'] >= lower_energy_bound]

	keff_test += [float(dftest['keff'].values[0])]
	xs_values = dftest['XS'].values

	XS_test.append(xs_values)


XS_test = np.array(XS_test)
keff_mean = np.mean(keff_test)
keff_std = np.std(keff_test)
y_test = zscore(keff_test)

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
print('Scaling test data...')
for column in tqdm.tqdm(scaling_matrix_xtest[le_bound_index:-1], total=len(scaling_matrix_xtest[le_bound_index:-1])):
	scaled_column = zscore(column)
	scaled_columns_xtest.append(scaled_column)

scaled_columns_xtest = np.array(scaled_columns_xtest)
X_test = scaled_columns_xtest.transpose()


callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=20,
										 mode='min',
										 start_from_epoch=3,
										 restore_best_weights=True)



model =keras.Sequential()
model.add(keras.layers.Dense(100, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(75, activation='relu'))
# model.add(keras.layers.Dense(400, activation='relu'))
# model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(40, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='MeanSquaredError', optimizer='adam')


import datetime
trainstart = time.time()
history = model.fit(X_train,
					y_train,
					epochs=75,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_test, y_test),
					verbose=1)

train_end = time.time()
print(f'Training completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
predictions = model.predict(X_test)
predictions = predictions.ravel()


rescaled_predictions = []
predictions_list = predictions.tolist()

for pred in predictions_list:
	descaled_p = pred * keff_std + keff_mean
	rescaled_predictions.append(float(descaled_p))

errors = []
for predicted, true in zip(rescaled_predictions, keff_test):
	errors.append((predicted - true) * 1e5)
	print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

sorted_errors = sorted(errors)
absolute_errors = [abs(x) for x in sorted_errors]
print(f'Average absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')

print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')


print(f"Smallest absolute error: {min(absolute_errors)} pcm")