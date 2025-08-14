import pandas as pd
# import tensorflow as tf
import keras
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os
import scipy.stats
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
















# model =keras.Sequential()
# model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))

