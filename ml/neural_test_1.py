import pandas as pd
# import tensorflow as tf
import keras
import numpy
import sklearn.preprocessing
import matplotlib.pyplot as plt
import os
import random


data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata'

all_csvs = os.listdir(data_directory)

training_csvs = []
while len(training_csvs) < 290:
	choice = random.choice(all_csvs)
	if choice not in training_csvs:
		training_csvs.append(choice)


X_train = []
y_train = []
for file in training_csvs:
	df = pd.read_csv(f'{data_directory}/{file}')
	break










# model =keras.Sequential()
# model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))

