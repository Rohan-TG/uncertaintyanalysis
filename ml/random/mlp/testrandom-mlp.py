import os
import sys

computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore
import tqdm






data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/Pu239/random/fission/xserg_data'

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

print('Fetching training data...')
keff_train = [] # k_eff labels
XS_train = []
for file in tqdm.tqdm(training_files, total=len(training_files)):
	# group = file.split('_')[1][1]

	dftrain = pd.read_parquet(f'{data_directory}/{file}', engine='pyarrow')

	keff_train += [float(dftrain['keff'].values[0])]  # append k_eff value from the file

	mt18xs = dftrain['XS'].values  # appends a list of fission cross sections to the XS_fission_train matrix
	# mt18xs = mt18xs[1:391]

	XS_train.append(mt18xs)

XS_train = np.array(XS_train)
y_train = zscore(keff_train)

scaling_matrix_xtrain = XS_train.transpose()

scaled_columns_xtrain = []
print('Scaling training data...')
for column in tqdm.tqdm(scaling_matrix_xtrain[1:-1], total=len(scaling_matrix_xtrain[1:-1])):
	scaled_column = zscore(column)
	scaled_columns_xtrain.append(scaled_column)

scaled_columns_xtrain = np.array(scaled_columns_xtrain)
X_train = scaled_columns_xtrain.transpose()


XS_test = []
keff_test = []

print('Fetching test data...')
for file in tqdm.tqdm(test_files, total=len(test_files)):
	# group = file.split('_')[1][1]
	dftest = pd.read_parquet(f'{data_directory}/{file}', engine='pyarrow')

	keff_test += [float(dftest['keff'].values[0])]
	xs_values = dftest['XS'].values

	XS_test.append(xs_values)


XS_test = np.array(XS_test)
keff_mean = np.mean(keff_test)
keff_std = np.std(keff_test)
y_test = zscore(keff_test)

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
print('Scaling test data...')
for column in tqdm.tqdm(scaling_matrix_xtest[1:-1], total=len(scaling_matrix_xtest[1:-1])):
	scaled_column = zscore(column)
	scaled_columns_xtest.append(scaled_column)

scaled_columns_xtest = np.array(scaled_columns_xtest)
X_test = scaled_columns_xtest.transpose()

