import pandas as pd
import keras
import numpy as np
import os
import time
from scipy.stats import zscore
import random
import tqdm
import matplotlib.pyplot as plt


data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/g4_fission_elastic_pu9'

all_parquets = os.listdir(data_directory)

n_training_samples = int(0.9 * len(all_parquets))

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_parquets)
	if choice not in training_files:
		training_files.append(choice)

test_files = []
for file in all_parquets:
	if file not in training_files:
		test_files.append(file)


y_train = [] # k_eff labels

XS_train = []

for file in tqdm.tqdm(training_files, total=len(training_files)):
	dftrain = pd.read_parquet(f'{data_directory}/{file}', engine='pyarrow') # Fetch data from parquet file

	y_train += [float(dftrain['keff'].values[0])] # append k_eff value from the file

	mt18xs = dftrain['MT18XS'].values # appends a list of fission cross sections to the XS_fission_train matrix

	mt2xs = dftrain['MT2XS'].values # likewise for elastic scattering cross sections



