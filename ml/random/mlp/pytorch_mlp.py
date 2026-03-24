import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import matplotlib.pyplot as plt
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie' or computer == 'bethe':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import random
import numpy as np
from scipy.stats import zscore
import tqdm
import time



print(f'\nGPU availability: {torch.cuda.is_available()}')


data_directory = input('Data directory: ')
data_processes = int(input('Num. data processors: '))

all_parquets = os.listdir(data_directory)

training_fraction = float(input('Enter training data fraction: '))
lower_energy_bound = float(input('Enter lower energy bound in eV: '))
patience = int(input('\nEnter patience: '))
n_training_samples = int(training_fraction * len(all_parquets))



training_files = []
while len(training_files) < n_training_samples:
	choice = random.choice(all_parquets)
	if choice not in training_files:
		training_files.append(choice)

val_files = []
for file in all_parquets:
	if file not in training_files:
		val_files.append(file)

print('\nFetching training data...')

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


for train_file in tqdm.tqdm(training_files, total=n_training_samples):
	xs_values, keff_value = fetch_data(train_file)

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


XS_val = []
keff_val = []

print('\nFetching val data...')


for val_file in tqdm.tqdm(val_files, total=len(val_files)):
	xs_values_val, keff_value_val = fetch_data(val_file)
	XS_val.append(xs_values_val)
	keff_val.append(keff_value_val)

XS_val = np.array(XS_val)
y_val = (np.array(keff_val) - train_labels_mean) / train_labels_std

scaling_matrix_xval = XS_val.transpose()

scaled_columns_xval = []
print('\nScaling val data...')
for column, c_mean, c_std in tqdm.tqdm(zip(scaling_matrix_xval[le_bound_index:-1], training_column_means, training_column_stds), total=len(scaling_matrix_xval[le_bound_index:-1])):
	# scaled_column = zscore(column)

	scaled_column = (np.array(column) - c_mean) / c_std
	scaled_columns_xval.append(scaled_column)

scaled_columns_xval = np.array(scaled_columns_xval)
X_val = scaled_columns_xval.transpose()

X_val = np.nan_to_num(X_val, nan=0.0)

X_train = np.nan_to_num(X_train, nan=0.0)








# Generate pytorch stuff

class MLP(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, 1000),  # Keras first Dense(1000), no activation
			nn.Linear(1000, 900),
			nn.ReLU(),
			nn.Linear(900, 750),
			nn.ReLU(),
			nn.Linear(750, 600),
			nn.ReLU(),
			nn.Linear(600, 540),
			nn.ReLU(),
			nn.Linear(540, 380),
			nn.ReLU(),
			nn.Linear(380, 280),
			nn.ReLU(),
			nn.Linear(280, 150),
			nn.ReLU(),
			nn.Linear(150, 100),
			nn.ReLU(),
			nn.Linear(100, 1)  # linear output
		)

		self._init_weights()

	def _init_weights(self):
		# Approximate Keras kernel_initializer='normal'
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean=0.0, std=0.05)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		return self.net(x)

class EarlyStopping:
	def __init__(self, patience=10, min_delta=0.0):
		self.patience = patience
		self.min_delta = min_delta
		self.best_loss = float("inf")
		self.counter = 0
		self.best_state = None
		self.early_stop = False

	def step(self, val_loss, model):
		if val_loss < self.best_loss - self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
			self.best_state = copy.deepcopy(model.state_dict())
		else:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()

# If your data is numpy arrays:
X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)

X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

# Define model
model = MLP(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

early_stopper = EarlyStopping(patience=patience)
num_epochs = 100

batch_size = 32
n = X_train_t.shape[0]

for epoch in range(num_epochs):
	model.train()

	perm = torch.randperm(n, device=device)
	epoch_loss = 0.0

	for i in range(0, n, batch_size):
		idx = perm[i:i+batch_size]
		xb = X_train_t[idx]
		yb = y_train_t[idx]

		optimizer.zero_grad()
		preds = model(xb)
		loss = criterion(preds, yb)
		loss.backward()
		optimizer.step()

		epoch_loss += loss.item() * xb.size(0)

	epoch_loss /= n

	model.eval()
	with torch.no_grad():
		val_loss = criterion(model(X_val_t), y_val_t).item()

	print(f"Epoch {epoch+1:3d} | train_loss={epoch_loss:.6f} | val_loss={val_loss:.6f}")

	stop = early_stopper.step(val_loss, model)
	if stop:
		print("Early stopping triggered.")
		break

# evaluate
model.eval()
with torch.no_grad():
	predictions = model(X_val_t)
	val_loss = criterion(predictions, y_val_t)

print("\nVal MSE:", val_loss.item())
print('\nProcessing results...')
rescaled_predictions = []
predictions_list = predictions.ravel()
predictions_list = predictions_list.tolist()
for pred in predictions_list:
	descaled_p = pred * train_labels_std + train_labels_mean
	rescaled_predictions.append(float(descaled_p))

errors = []
for predicted, true in zip(rescaled_predictions, keff_val):
	errors.append((predicted - true) * 1e5)
	# print(f'SCONE: {true:0.5f} - ML: {predicted:0.5f}, Difference = {(predicted - true) * 1e5:0.0f} pcm')

# Save data into matrices
# for p_index, p in enumerate(rescaled_predictions):
# 	prediction_matrix[p_index].append(p)
#
# for err_index, err in enumerate(errors):
# 	error_matrix[err_index].append(err)

sorted_errors = sorted(errors)
absolute_errors = [abs(x) for x in sorted_errors]
print(f'\nAverage absolute error: {np.mean(absolute_errors)} +- {np.std(absolute_errors)}')

print(f'Max -ve error: {sorted_errors[0]} pcm, Max +ve error: {sorted_errors[-1]} pcm')


print(f"Smallest absolute error: {min(absolute_errors)} pcm")
acceptable_predictions = []
borderline_predictions = []
fifteen_pcm_predictions = []
twenty_pcm_predictions = []
for x in absolute_errors:
	if x <= 5.0:
		acceptable_predictions.append(x)
	if x <= 10.0:
		borderline_predictions.append(x)
	if x <= 15.0 :
		fifteen_pcm_predictions.append(x)
	if x <= 20.0:
		twenty_pcm_predictions.append(x)


print(f' {len(acceptable_predictions)} ({len(acceptable_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 5 pcm error')
print(f' {len(borderline_predictions)} ({len(borderline_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 10 pcm error')
print(f' {len(fifteen_pcm_predictions)} ({len(fifteen_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 15 pcm error)')
print(f' {len(twenty_pcm_predictions)} ({len(twenty_pcm_predictions) / len(absolute_errors) * 100:.2f}%) predictions <= 20 pcm error)')
