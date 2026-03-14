import os
num_threads = 30
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_threads}"
os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_threads}"
os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_threads}"
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# MLP

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
import keras
import time
import keras_tuner as kt


print('\n')
data_directory = input('Data directory: ')
test_data_directory = input('\nTest data directory (x for set to val): ')


data_processes = 5

all_parquets = os.listdir(data_directory)

training_fraction = float(input('\nEnter training data fraction: '))
lower_energy_bound = float(input('\nEnter lower energy bound in eV: '))
patience = int(input('\nPatience: '))

try:
	mask = float(input('\nMask (x skip): '))
except:
	mask = 'x'
	print('Skip masking...')

n_training_samples = int(training_fraction * len(all_parquets))


print('\nFetching training data...')

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_parquets)
	if choice not in training_files:
		training_files.append(choice)

test_files = []
for file in all_parquets:
	if file not in training_files:
		test_files.append(file)



def fetch_data(datafile):

	temp_df = pd.read_parquet(f'{data_directory}/{datafile}', engine='pyarrow')
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

	# xsobject = pu9_mt2xs + pu9_mt4xs +  pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt18xs + pu1_mt102xs
	xsobject = pu9_mt2xs + pu9_mt4xs + pu9_mt16xs + pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt16xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt16xs + pu1_mt18xs + pu1_mt102xs

	XS_obj = xsobject

	return(XS_obj, keff_value)



keff_train = [] # k_eff labels
XS_train = []

with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, train_file) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, keff_value = future.result()
		XS_train.append(xs_values)
		keff_train.append(keff_value)

XS_train = np.array(XS_train)
y_train = zscore(keff_train)

train_labels_mean = np.mean(keff_train)
train_labels_std = np.std(keff_train)


scaling_matrix_xtrain = XS_train.transpose()

scaled_columns_xtrain = []
print('\nScaling training data...')


le_bound_index = 1 # filters out NaNs


training_column_means = []
training_column_stds = []

for column in tqdm.tqdm(scaling_matrix_xtrain[le_bound_index:-1], total=len(scaling_matrix_xtrain[le_bound_index:-1])):
	scaled_column = zscore(column)

	column_mean = np.mean(column)
	column_std = np.std(column)
	training_column_means.append(column_mean)
	training_column_stds.append(column_std)
	scaled_columns_xtrain.append(scaled_column)

scaled_columns_xtrain = np.array(scaled_columns_xtrain)
X_train = scaled_columns_xtrain.transpose()


XS_test = []
keff_test = []

print('\nFetching test data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, test_file) for test_file in test_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_test, keff_value_test = future.result()
		XS_test.append(xs_values_test)
		keff_test.append(keff_value_test)

XS_test = np.array(XS_test)
# keff_mean = np.mean(keff_test)
# keff_std = np.std(keff_test)
y_test = (np.array(keff_test) - train_labels_mean) / train_labels_std

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
print('\nScaling test data...')
for column, c_mean, c_std in tqdm.tqdm(zip(scaling_matrix_xtest[le_bound_index:-1], training_column_means, training_column_stds), total=len(scaling_matrix_xtest[le_bound_index:-1])):
	# scaled_column = zscore(column)

	scaled_column = (np.array(column) - c_mean) / c_std
	scaled_columns_xtest.append(scaled_column)

scaled_columns_xtest = np.array(scaled_columns_xtest)
X_test = scaled_columns_xtest.transpose()


# test_mask = ~np.isnan(X_test).any(axis=0)
# X_test = X_test[:, test_mask]
X_test = np.nan_to_num(X_test, nan=0.0)

# train_mask = ~np.isnan(X_train).any(axis=0)
# X_train = X_train[:, train_mask]
X_train = np.nan_to_num(X_train, nan=0.0)





class tunerHyperModel(kt.HyperModel):

	def build(self, hp):

		hp_units = hp.Int('input_nodes', min_value=100, max_value=2000, step=200)

		model = keras.Sequential()
		model.add(keras.layers.Dense(hp_units, input_shape=(X_train.shape[1],), kernel_initializer='normal'))

		n_dense_1 = hp.Int("n_dense_layers", min_value=1, max_value=5, step=1)  # varies number of layers used
		for n_layers in range(n_dense_1):
			node_units_1 = hp.Int(f'dense_{n_layers}_units', min_value=500, max_value=1800, step=100)
			model.add(keras.layers.Dense(node_units_1, activation='relu'))

		n_dense_2 = hp.Int('n_dense_layers_2', min_value=1, max_value=5, step=1)
		for n_layers_2 in range(n_dense_2):
			node_units_2 = hp.Int(f'dense_{n_layers_2}_units', min_value=10, max_value=800, step=100)
			model.add(keras.layers.Dense(node_units_2, activation='relu'))

		model.add(keras.layers.Dense(1, activation='linear'))
		model.compile(loss='MeanSquaredError', optimizer='adam')

		print(model.summary())

		return model

tuner = kt.BayesianOptimization(hypermodel=tunerHyperModel(),
								objective='val_loss',
								max_trials=200,
								)

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=patience,
										 mode='min',
										 start_from_epoch=3,
										 restore_best_weights=True)

tuner.search(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=2000,
    batch_size=32,
	callbacks = [callback]
)

best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

predictions = best_model.predict(X_test)
predictions = predictions.ravel()


rescaled_predictions = []
predictions_list = predictions.tolist()

for pred in predictions_list:
	descaled_p = pred * train_labels_std + train_labels_mean
	rescaled_predictions.append(float(descaled_p))

errors = []
for predicted, true in zip(rescaled_predictions, keff_test):
	errors.append((predicted - true) * 1e5)
	print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

sorted_errors = sorted(errors)
absolute_errors = [abs(x) for x in sorted_errors]
print(f'\nAverage absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')

print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')


print(f"Smallest absolute error: {min(absolute_errors)} pcm")
acceptable_predictions = []
borderline_predictions = []
twenty_pcm_predictions = []
for x in absolute_errors:
	if x <= 5.0:
		acceptable_predictions.append(x)
	if x <= 10.0:
		borderline_predictions.append(x)
	if x <= 20.0:
		twenty_pcm_predictions.append(x)


print(f' {len(acceptable_predictions)} ({len(acceptable_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 5 pcm error')
print(f' {len(borderline_predictions)} ({len(borderline_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 10 pcm error')
print(f' {len(twenty_pcm_predictions)} ({len(twenty_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 20 pcm error)')


savename = input('Model save name: ')
best_model.save(f'{savename}.keras')