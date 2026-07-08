import os
import matplotlib.pyplot as plt
import keras.backend
import pickle
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore, norm
import tqdm
import keras
import time
import tensorflow as tf
import datetime


model_directory = input('\nModel directory: ')

data_directory = input('\n\nData directory: ')
test_directory = input('\nTest directory (x set to val): ')
if test_directory != 'x':
	test_files = os.listdir(test_directory)



all_parquets = os.listdir(data_directory)
lower_energy_bound = float(input('\nEnter lower energy bound in eV: '))