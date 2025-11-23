import pandas as pd
# import tensorflow as tf
import keras
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os
import time
from scipy.stats import zscore
import random
import tqdm


data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/g1fissiondata/combined'

all_csvs = os.listdir(data_directory)

training_fraction = float(input('Enter training data fraction: '))

n_training_samples = int(training_fraction * len(all_csvs))

training_csvs = []
while len(training_csvs) < n_training_samples:
	choice = random.choice(all_csvs)
	if choice not in training_csvs:
		training_csvs.append(choice)

test_csvs = []
for csv in all_csvs:
	if csv not in training_csvs:
		test_csvs.append(csv)

maxrow = 188601
XS_train = []
y_train = []
for file in tqdm.tqdm(training_csvs, total=len(training_csvs)):
	df = pd.read_csv(f'{data_directory}/{file}')
	y_train += [float(df['keff'].values[0])]
	XS_train.append(df['XS'].values[:maxrow])

XS_train = np.array(XS_train)
y_train = zscore(y_train)

# NOTE: X_train[:,n] means all samples (:) and the nth energy point for each sample
scaling_matrix_xtrain = XS_train.transpose()



scaled_columns_xtrain = []
for column in tqdm.tqdm(scaling_matrix_xtrain[1:], total=len(scaling_matrix_xtrain[1:])):
	scaled_column = zscore(column)
	scaled_columns_xtrain.append(scaled_column)

scaled_columns_xtrain = np.array(scaled_columns_xtrain)
X_train = scaled_columns_xtrain.transpose()




XS_test = []
keff_test = []
for file in tqdm.tqdm(test_csvs, total=len(test_csvs)):
	dftest = pd.read_csv(f'{data_directory}/{file}')
	keff_test += [float(dftest['keff'].values[0])]
	XS_test.append(dftest['XS'].values[:maxrow])

XS_test = np.array(XS_test)
keff_mean = np.mean(keff_test)
keff_std = np.std(keff_test)
y_test = zscore(keff_test)

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
for column in tqdm.tqdm(scaling_matrix_xtest[1:], total=len(scaling_matrix_xtest[1:])):
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


model = keras.Sequential()
model.add(keras.layers.Input(shape=(None, 1, X_train.shape[1])))
model.add(keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu',))
model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(10, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
# # model.add(keras.layers.Dense(1000, activation='relu'))
# # model.add(keras.layers.Dense(500, activation='relu'))
# model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='MeanSquaredError', optimizer='adam')

import datetime
trainstart = time.time()
history = model.fit(X_train,
					y_train,
					epochs=100,
					batch_size=16,
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