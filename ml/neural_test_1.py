import pandas as pd
# import tensorflow as tf
import keras
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore
import random
import tqdm


data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata'

all_csvs = os.listdir(data_directory)

training_csvs = []
while len(training_csvs) < 290:
	choice = random.choice(all_csvs)
	if choice not in training_csvs:
		training_csvs.append(choice)

test_csvs = []
for csv in all_csvs:
	if csv not in training_csvs:
		test_csvs.append(csv)

XS_train = []
y_train = []
for file in tqdm.tqdm(training_csvs, total=len(training_csvs)):
	df = pd.read_csv(f'{data_directory}/{file}')
	y_train += [float(df['keff'].values[0])]
	XS_train.append(df['XS'].values)

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
y_test = []
for file in tqdm.tqdm(test_csvs, total=len(test_csvs)):
	dftest = pd.read_csv(f'{data_directory}/{file}')
	y_test += [float(dftest['keff'].values[0])]
	XS_test.append(dftest['XS'].values)

XS_test = np.array(XS_test)
y_test = zscore(y_test)

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
for column in tqdm.tqdm(scaling_matrix_xtest[1:], total=len(scaling_matrix_xtest[1:])):
	scaled_column = zscore(column)
	scaled_columns_xtest.append(scaled_column)

scaled_columns_xtest = np.array(scaled_columns_xtest)
X_test = scaled_columns_xtest.transpose()










model =keras.Sequential()
model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')

history = model.fit(X_train,
					y_train,
					epochs=20,
					batch_size=16,
					# callbacks=callback,
					validation_data=(X_test, y_test),
					verbose=1)


predictions = model.predict(X_test)
predictions = predictions.ravel()