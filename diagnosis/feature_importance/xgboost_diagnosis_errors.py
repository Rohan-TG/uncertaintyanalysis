import os
import tqdm
import xgboost as xg
import pandas as pd
import random
import numpy as np


error_data_directory = input('Error data directory: ')

all_files = os.listdir(error_data_directory)

training_fraction = float(input('Training fraction: '))
n_training_samples = int(training_fraction * len(all_files))

lower_energy_bound = float(input('Lower energy bound eV: '))

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_files)
	if choice not in training_files:
		training_files.append(choice)


validation_files = []
for f in all_files:
	if f not in training_files:
		validation_files.append(f)


def fetch_data(datafile, data_dir=error_data_directory):

	temp_df = pd.read_parquet(f'{data_dir}/{datafile}', engine='pyarrow')
	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]

	keff_value = float(temp_df['keff'].values[0])

	pu9_mt18xs = temp_df['94239_MT18_XS'].values.tolist()
	pu0_mt18xs = temp_df['94240_MT18_XS'].values.tolist()
	pu1_mt18xs = temp_df['94241_MT18_XS'].values.tolist()

	pu9_mt2xs = temp_df['94239_MT2_XS'].values.tolist()
	pu0_mt2xs = temp_df['94240_MT2_XS'].values.tolist()
	pu1_mt2xs = temp_df['94241_MT2_XS'].values.tolist()

	pu9_mt4xs = temp_df['94239_MT4_XS'].values.tolist()
	pu0_mt4xs = temp_df['94240_MT4_XS'].values.tolist()
	pu1_mt4xs = temp_df['94241_MT4_XS'].values.tolist()

	pu9_mt16xs = temp_df['94239_MT16_XS'].values.tolist()
	pu0_mt16xs = temp_df['94240_MT16_XS'].values.tolist()
	pu1_mt16xs = temp_df['94241_MT16_XS'].values.tolist()

	pu9_mt102xs = temp_df['94239_MT102_XS'].values.tolist()
	pu0_mt102xs = temp_df['94240_MT102_XS'].values.tolist()
	pu1_mt102xs = temp_df['94241_MT102_XS'].values.tolist()

	XS_obj = [pu9_mt2xs, pu9_mt4xs, pu9_mt16xs, pu9_mt18xs, pu9_mt102xs,
				pu0_mt2xs, pu0_mt4xs, pu0_mt16xs, pu0_mt18xs, pu0_mt102xs,
				pu1_mt2xs, pu1_mt4xs, pu1_mt16xs, pu1_mt18xs, pu1_mt102xs,]

	return(XS_obj, keff_value)


channel_keys = {'94239_MT2_XS': 0,'94239_MT4_XS': 1,'94239_MT16_XS': 2,
				'94239_MT18_XS': 3,'94239_MT102_XS': 4,'94240_MT2_XS': 5,
				'94240_MT4_XS': 6,'94240_MT16_XS': 7,'94240_MT18_XS': 8,
				'94240_MT102_XS': 9,'94241_MT2_XS': 10,'94241_MT4_XS': 11,
				'94241_MT16_XS': 12,'94241_MT18_XS': 13,'94241_MT102_XS': 14,}


print('\nFetching training data...\n')


raw_train = []
y_train = []
for train_f in tqdm.tqdm(training_files, total=len(training_files)):
	xs_obj_train, labels_train = fetch_data(train_f)

	raw_train.append(xs_obj_train)
	y_train.append(labels_train)


print('\nFetching validation data...\n')

raw_val = []
y_val = []
for val_f in tqdm.tqdm(validation_files, total=len(validation_files)):
	xs_obj_val, labels_val = fetch_data(val_f)

	raw_val.append(xs_obj_val)
	y_val.append(labels_val)




def make_X_matrix(matrix):
	channel_columns = [[] for i in matrix[0]]

	for sample in tqdm.tqdm(matrix, total=len(matrix)):
		for j, channel in enumerate(sample):
			channel_columns[j].append(channel)

	X_matrix = []
	for channel in channel_columns:
		flattened_channel = np.array(channel).ravel()
		X_matrix.append(flattened_channel)

	return np.array(X_matrix)



print('\nMaking X_train...')
X_train = make_X_matrix(raw_train)

print('\nMaking X_val...')
X_val = make_X_matrix(raw_val)




model = xg.XGBRegressor(n_estimators=950, # define regressor
						learning_rate=0.01,
						max_depth=8,
						subsample=0.18236,
						max_leaves=0,
						seed=42, )


model.fit(X_train, y_train, verbose=True,
		  eval_set = [(X_val, y_val)])

print("\nTraining complete")

predictions = model.predict(X_val)

for error, predicted_error in zip(y_val, predictions):
	print(f'Error: {error} --- Predicted error: {predicted_error} --- Difference: {error - predicted_error}')
