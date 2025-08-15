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


X_train = []
y_train = []
for file in tqdm.tqdm(training_csvs, total=len(training_csvs)):
	df = pd.read_csv(f'{data_directory}/{file}')
	y_train += [float(df['keff'].values[0])]
	X_train.append(df['XS'].values)

X_train = np.array(X_train)


# NOTE: X_train[:,n] means all samples (:) and the nth energy point for each sample
scaling_matrix_xtrain = X_train.transpose()


scaled_columns_xtrain = []
for column in tqdm.tqdm(scaling_matrix_xtrain, total=len(scaling_matrix_xtrain)):
	scaled_column = zscore(column)
	scaled_columns_xtrain.append(scaled_column)














# model =keras.Sequential()
# model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))

