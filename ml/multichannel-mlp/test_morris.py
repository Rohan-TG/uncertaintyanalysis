import pandas as pd
import keras
import numpy as np
import os
import time
from scipy.stats import zscore
import random
import tqdm
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')
from groupEnergies import Groups
training_fission_data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/Pu239/fission/g4'
training_elastic_data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/Pu239/elastic_scattering/g4'
test_data_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/g4_fission_elastic_pu9'

g4boundary = Groups.g4
g3boundary = Groups.g3

all_fission_parquets = [os.path.join(training_fission_data_directory, f) for f in os.listdir(training_fission_data_directory)]
all_elastic_parquets = [os.path.join(training_elastic_data_directory, f) for f in os.listdir(training_elastic_data_directory)]

perturbation_set = np.arange(-0.500, 0.501, 0.001)

training_fraction = 1.0
n_training_samples = int(training_fraction * len(perturbation_set))

# while len(training_files) < n_training_samples:
# 	choice = random.choice(perturbation_set)
# 	if choice not in training_files:
# 		training_files.append(choice)

test_files = os.listdir(test_data_directory)
# # for file in all_parquets:
# # 	if file not in training_files:
# # 		test_files.append(file)


keff_train = [] # k_eff labels

XS_train = []
fission_train = []
elastic_train = []

for file in tqdm.tqdm(all_fission_parquets, total=len(all_fission_parquets)):
	dftrain = pd.read_parquet(f'{file}', engine='pyarrow') # Fetch data from parquet file

	keff_train += [float(dftrain['keff'].values[0])] # append k_eff value from the file

	dftrain = dftrain[dftrain.ERG >= g4boundary]
	dftrain = dftrain[dftrain.ERG <= g3boundary]

	mt18xs = dftrain['XS'].values # appends a list of cross sections to the XS_train matrix

	elastic_df = pd.read_parquet('/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/Pu239/elastic_scattering/g4/Pu-239_g4_0.000_MT2.parquet',
								 engine='pyarrow')

	elastic_df = elastic_df[elastic_df.ERG >= g4boundary]
	elastic_df = elastic_df[elastic_df.ERG <= g3boundary]

	mt2xs = elastic_df['XS'].values # likewise for elastic scattering cross sections

	fission_train.append(mt18xs)
	elastic_train.append(mt2xs)

for file in tqdm.tqdm(all_elastic_parquets, total=len(all_elastic_parquets)):
	dftrain = pd.read_parquet(f'{file}', engine='pyarrow')  # Fetch data from parquet file

	keff_train += [float(dftrain['keff'].values[0])]  # append k_eff value from the file

	dftrain = dftrain[dftrain.ERG >= g4boundary]
	dftrain = dftrain[dftrain.ERG <= g3boundary]

	mt2xs = np.log(dftrain['XS'].values)  # appends a list of cross sections to the XS_train matrix

	fission_df = pd.read_parquet(
		'/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata/Pu239/fission/g4/Pu-239_g4_0.000_MT18.parquet',
		engine='pyarrow')

	fission_df = fission_df[fission_df.ERG >= g4boundary]
	fission_df = fission_df[fission_df.ERG <= g3boundary]

	mt18xs = np.log(fission_df['XS'].values)  # likewise for elastic scattering cross sections

	fission_train.append(mt18xs)
	elastic_train.append(mt2xs)


y_train = np.array(keff_train)
y_train = zscore(y_train)

keff_train_mean = np.mean(keff_train)
keff_train_std = np.std(keff_train)











def scaler(channel_matrix):

	transposed_matrix = np.transpose(np.array(channel_matrix))
	scaled_rows = []
	for row in tqdm.tqdm(transposed_matrix, total=len(transposed_matrix)):
		scaled_row = zscore(row)
		scaled_rows.append(scaled_row)

	scaled_rows = np.array(scaled_rows)
	transposed_scaled_matrix = np.transpose(scaled_rows)

	final_matrix = []
	for i in transposed_scaled_matrix:
		row = i[~np.isnan(i)]
		final_matrix.append(row)

	final_matrix = np.array(final_matrix)
	return final_matrix



scaled_fission_train = scaler(fission_train)
scaled_elastic_train = scaler(elastic_train)


X_train = []
for f_data, e_data in zip(scaled_fission_train, scaled_elastic_train):
	X_train.append([f_data, e_data])

X_train = np.array(X_train)
X_train = X_train.reshape([X_train.shape[0], X_train.shape[2], X_train.shape[1]])


print('Training data processed...')






########################################## Test data preparation #######################################################
keff_test = []

fission_test = []
elastic_test = []
for testfile in tqdm.tqdm(test_files, total=len(test_files)):
	dftest = pd.read_parquet(f'{test_data_directory}/{testfile}', engine='pyarrow')
	keff_test += [float(dftest['keff'].values[0])]

	dftest = dftest[dftest.ERG >= g4boundary]
	dftest = dftest[dftest.ERG <= g3boundary]

	mt18xstest = dftest['MT18_XS'].values  # appends a list of fission cross sections to the XS_fission_train matrix

	mt2xstest = dftest['MT2_XS'].values  # likewise for elastic scattering cross sections

	fission_test.append(mt18xstest)
	elastic_test.append(mt2xstest)


y_test = np.array(keff_test)
keff_mean = np.mean(keff_test)
keff_std = np.std(keff_test)
y_test = zscore(y_test)

scaled_fission_test = scaler(fission_test)
scaled_elastic_test = scaler(elastic_test)
X_test = []

for f_data, e_data in zip(scaled_fission_test, scaled_elastic_test):
	X_test.append([f_data, e_data])

X_test = np.array(X_test)
X_test = X_test.reshape([X_test.shape[0], X_test.shape[2], X_test.shape[1]])
print('Test data processed...')

