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

training_fraction = float(input('Enter training data fraction: '))
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


fission_train = []
elastic_train = []

for file in tqdm.tqdm(training_files, total=len(training_files)):
	dftrain = pd.read_parquet(f'{data_directory}/{file}', engine='pyarrow') # Fetch data from parquet file

	keff_train += [float(dftrain['keff'].values[0])] # append k_eff value from the file

	dftrain = dftrain[dftrain.ERG >= g4boundary]
	dftrain = dftrain[dftrain.ERG <= g3boundary]

	mt18xs = dftrain['MT18_XS'].values # appends a list of fission cross sections to the XS_fission_train matrix

	mt2xs = dftrain['MT2_XS'].values # likewise for elastic scattering cross sections


	fission_train.append(mt18xs)
	elastic_train.append(mt2xs)


y_train = np.array(keff_train)
y_train = zscore(y_train)

keff_train_mean = np.mean(keff_train)
keff_train_std = np.std(keff_train)


def scaler(channel_matrix):

	transposed_matrix = np.transpose(np.array(channel_matrix))
	scaled_rows = []
	for row in tqdm.tqdm(transposed_matrix, total=len(transposed_matrix)):
		scaled_row = zscore(row)
		scaled_rows.append(scaled_row)

	scaled_rows = np.array(scaled_rows)
	transposed_scaled_matrix = np.transpose(scaled_rows)

	final_matrix = []
	for i in transposed_scaled_matrix:
		row = i[~np.isnan(i)]
		final_matrix.append(row)

	final_matrix = np.array(final_matrix)
	return final_matrix





scaled_fission_train = scaler(fission_train)
scaled_elastic_train = scaler(elastic_train)

X_train = []
for f_data, e_data in zip(scaled_fission_train, scaled_elastic_train):
	X_train.append([f_data, e_data])

X_train = np.array(X_train)
X_train = X_train.reshape([X_train.shape[0], X_train.shape[2], X_train.shape[1]])

print('Training data processed...')


########################################## Test data preparation #######################################################
keff_test = []

fission_test = []
elastic_test = []
for testfile in tqdm.tqdm(test_files, total=len(test_files)):
	dftest = pd.read_parquet(f'{data_directory}/{testfile}', engine='pyarrow')
	keff_test += [float(dftest['keff'].values[0])]

	dftest = dftest[dftest.ERG >= g4boundary]
	dftest = dftest[dftest.ERG <= g3boundary]

	mt18xstest = dftest['MT18_XS'].values  # appends a list of fission cross sections to the XS_fission_train matrix

	mt2xstest = dftest['MT2_XS'].values  # likewise for elastic scattering cross sections

	fission_test.append(mt18xstest)
	elastic_test.append(mt2xstest)


y_test = np.array(keff_test)
keff_mean = np.mean(keff_test)
keff_std = np.std(keff_test)
y_test = zscore(y_test)

scaled_fission_test = scaler(fission_test)
scaled_elastic_test = scaler(elastic_test)
X_test = []

for f_data, e_data in zip(scaled_fission_test, scaled_elastic_test):
	X_test.append([f_data, e_data])

X_test = np.array(X_test)
X_test = X_test.reshape([X_test.shape[0], X_test.shape[2], X_test.shape[1]])
print('Test data processed...')


callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=20,
										 mode='min',
										 start_from_epoch=5,
										 restore_best_weights=True)

model = keras.Sequential()
model.add(keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu',))
model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(10, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
# # model.add(keras.layers.Dense(1000, activation='relu'))
# # model.add(keras.layers.Dense(500, activation='relu'))
# model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='MeanSquaredError', optimizer='adam')


trainstart = time.time()
history = model.fit(X_train,
					y_train,
					epochs=100,
					batch_size=4,
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