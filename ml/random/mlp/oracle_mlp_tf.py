import gc
import os
import matplotlib.pyplot as plt
import keras.backend
import pickle
# MLP
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie' or computer == 'bethe':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore, norm
import tqdm
import keras
import time
import tensorflow as tf
import datetime

print(tf.config.list_physical_devices('GPU'))


data_directory = input('\n\nData directory: ')

all_parquets = os.listdir(data_directory)


errors_directory = input('\nErrors directory: ')

predictions_directory = input('\nPredictions directory: ')

