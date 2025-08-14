import pandas as pd
import tensorflow as tf
import keras
import numpy
import sklearn.preprocessing
import matplotlib.pyplot as plt




model =keras.Sequential()
model.add(keras.layers.Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='normal'))

