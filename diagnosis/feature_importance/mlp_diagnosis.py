import os
import sys
import matplotlib.pyplot as plt
# import shap

# MLP

# computer = os.uname().nodename
# if computer == 'fermiac':
# 	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
# elif computer == 'oppie':
# 	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore
import tqdm
import keras
import time



error_data_directory = input('Error data directory: ')

all_files = os.listdir(error_data_directory)

training_fraction = float(input('Training fraction: '))
n_training_samples = int(training_fraction * len(all_files))

lower_energy_bound = float(input('Lower energy bound eV: '))

patience = int(input('Patience: '))

training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_files)
	if choice not in training_files:
		training_files.append(choice)

test_files = []
for file in all_files:
	if file not in training_files:
		test_files.append(file)

print('\nFetching training data...')





def fetch_data(datafile):

	temp_df = pd.read_parquet(f'{error_data_directory}/{datafile}', engine='pyarrow')
	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]

	error_value = temp_df['ml_error'].values[0]

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

	xsobject = pu9_mt2xs + pu9_mt4xs + pu9_mt16xs + pu9_mt18xs + pu9_mt18xs + pu9_mt102xs + pu0_mt2xs + pu0_mt4xs + pu0_mt16xs + pu0_mt18xs + pu0_mt102xs + pu1_mt2xs + pu1_mt2xs + pu1_mt4xs + pu1_mt16xs + pu1_mt18xs + pu1_mt102xs

	XS_obj = xsobject

	return(XS_obj, error_value)



error_train = [] # error labels
XS_train = []

for train_file in tqdm.tqdm(training_files, total=n_training_samples):
	xs_values, error_value = fetch_data(train_file)

	XS_train.append(xs_values)
	error_train.append(error_value)

XS_train = np.array(XS_train)
y_train = zscore(error_train)

train_labels_mean = np.mean(error_train)
train_labels_std = np.std(error_train)


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
error_test = []

print('\nFetching test data...')


for test_file in tqdm.tqdm(test_files, total=len(test_files)):
	xs_values_test, error_value_test = fetch_data(test_file)
	XS_test.append(xs_values_test)
	error_test.append(error_value_test)

XS_test = np.array(XS_test)
y_test = (np.array(error_test) - train_labels_mean) / train_labels_std

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


callback = keras.callbacks.EarlyStopping(monitor='val_loss',
										 # min_delta=0.005,
										 patience=patience,
										 mode='min',
										 restore_best_weights=True)




model =keras.Sequential()
model.add(keras.layers.Dense(1000, input_shape=(X_train.shape[1],), kernel_initializer='normal'))
model.add(keras.layers.Dense(900, activation='relu'))
model.add(keras.layers.Dense(750, activation='relu'))
model.add(keras.layers.Dense(600, activation='relu'))
model.add(keras.layers.Dense(540, activation='relu'))
model.add(keras.layers.Dense(380, activation='relu'))
model.add(keras.layers.Dense(280, activation='relu'))
model.add(keras.layers.Dense(150, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.compile(loss='MeanSquaredError', optimizer='adam')




import datetime
trainstart = time.time()
history = model.fit(X_train,
					y_train,
					epochs=3000,
					batch_size=32,
					callbacks=callback,
					validation_data=(X_test, y_test),
					verbose=1)

train_end = time.time()
print(f'\nTraining completed in {datetime.timedelta(seconds=(train_end - trainstart))}')
predictions = model.predict(X_test)
predictions = predictions.ravel()


rescaled_predictions = []
predictions_list = predictions.tolist()

for pred in predictions_list:
	descaled_p = pred * train_labels_std + train_labels_mean
	rescaled_predictions.append(float(descaled_p))

errors = []
for predicted, true in zip(rescaled_predictions, error_test):
	errors.append((predicted - true))
	print(f'ML Target: {true:0.5f} pcm - : Predicted value: {predicted:0.5f} pcm, Loss = {(predicted - true):0.0f} pcm')

sorted_errors = sorted(errors)
absolute_errors = [abs(x) for x in sorted_errors]
print(f'\nAverage absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')

print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')


print(f"Smallest absolute error: {min(absolute_errors)} pcm")
acceptable_predictions = []
borderline_predictions = []
for x in absolute_errors:
	if x <= 5.0:
		acceptable_predictions.append(x)
	if x <= 10.0:
		borderline_predictions.append(x)


print(f' {len(acceptable_predictions)} ({len(acceptable_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 5 pcm error')
print(f' {len(borderline_predictions)} ({len(borderline_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 10 pcm error')


plt.figure()
plt.hist(sorted_errors, bins=30)
plt.grid()
plt.title('Distribution of errors')
plt.xlabel('Error / pcm')
plt.ylabel('Count')
# plt.savefig('absolute_ml_errors_errors_corrected_scaling.png', dpi = 300)
plt.show()




skew_positive = []
skew_negative = []

for x in errors:
	if x >0:
		skew_positive.append(x)
	else:
		skew_negative.append(x)

plt.figure()
plt.plot(error_test, errors, 'x')
plt.grid()
plt.title('Distribution of errors')
plt.xlabel('True error')
plt.ylabel('Error / pcm')
# plt.savefig('errors_as_function_of_true_error.png', dpi = 300)
plt.show()

### Feature importance

# shap_values = shap.DeepExplainer(model=model, data=X_test)

