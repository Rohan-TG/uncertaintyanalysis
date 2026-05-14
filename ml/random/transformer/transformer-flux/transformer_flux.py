import pandas as pd
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
# from torch.utils.data import DataLoader, Dataset

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


xs_directory = input('XS directory: ')
test_data_directory = input('Test data directory (x for set to val): ')
max_epochs = int(input('Max epochs: '))
patience = int(input('Patience: '))
flux_data_directory = input('Flux data directory: ')


# data_processes = int(input('Num. data processors: '))
data_processes = 6
all_parquets = os.listdir(xs_directory)

training_fraction = float(input('Enter training data fraction: '))
lower_energy_bound = float(input('Enter lower energy bound in eV: '))

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



print('Fetching training data...')





def fetch_data(datafile):

	temp_df = pd.read_parquet(f'{xs_directory}/{datafile}', engine='pyarrow')
	temp_df = temp_df[temp_df['ERG'] >= lower_energy_bound]

	pu9_index = int(datafile.split('_')[4])
	pu0_index = int(datafile.split('_')[6])
	pu1_index = int(datafile.split('_')[8].split('.')[0])


	# keff_value = float(temp_df['keff'].values[0])

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


	# Now fetch spectrum data labels

	flux_file = f'Flux_data_Pu-239_{pu9_index}_Pu-240_{pu0_index}_Pu-241_{pu1_index}.parquet'
	flux_read_obj = pd.read_parquet(f'{flux_data_directory}/{flux_file}', engine='pyarrow')
	flux_data = flux_read_obj['flux'].values
	flux_error = flux_read_obj['flux_errror']

	pct_flux_error = np.array(flux_error) / np.array(flux_data)

	global flux_lower_bounds, flux_upper_bounds
	flux_lower_bounds = flux_read_obj['low_erg_bounds'].values
	flux_upper_bounds = flux_read_obj['high_erg_bounds'].values

	return(XS_obj,
		   flux_data,
		   flux_error,
		   )



flux_train = [] # flux labels
flux_train_error = []
XS_train = []


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, train_file) for train_file in training_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values, flux_values, flux_error = future.result()
		XS_train.append(xs_values)
		flux_train.append(flux_values)
		flux_train_error.append(flux_error)

XS_train = np.array(XS_train) # shape (num_samples, num_channels, points per channel)
flux_train = np.array(flux_train) # matrix
flux_train_error = np.array(flux_train_error)


def scale_flux(flux_array, flux_error_array, train_mode = False, means = None, stds = None):
	"""setting train_mode to True just makes this function return the means and stds. Otherwise not returned"""

	normalised_flux_array = []
	normalised_flux_error_array = []
	for flux_set, flux_error_set in zip(flux_array, flux_error_array):
		area = np.sum(flux_set)
		norm_flux_vector = np.array(flux_set) / area
		norm_flux_error = np.array(flux_error_set) / area
		normalised_flux_array.append(norm_flux_vector)
		normalised_flux_error_array.append(norm_flux_error)

	normalised_flux_array = np.array(normalised_flux_array)
	normalised_flux_error_array = np.array(normalised_flux_error_array)

	transposed_flux_array = normalised_flux_array.transpose()
	if train_mode:
		scaling_columns = []
		scaling_column_means = []
		scaling_column_stds = []
		for energy_column in transposed_flux_array:
			scaling_columns.append(zscore(energy_column))
			scaling_column_means.append(np.mean(energy_column))
			scaling_column_stds.append(np.std(energy_column))

		scaling_column_stds = np.array(scaling_column_stds)
		scaling_column_means = np.array(scaling_column_means)
		scaled_transposed_flux_array = np.array(scaling_columns)
		scaled_flux_array = scaled_transposed_flux_array.transpose()
		return scaled_flux_array, scaling_column_means, scaling_column_stds, normalised_flux_error_array

	else:
		scaling_columns = []
		for energy_column, mean, std in zip(transposed_flux_array, means, stds):
			scaling_columns.append((np.array(energy_column) - mean) / std)

		scaled_transposed_flux_array = np.array(scaling_columns)
		scaled_flux_array = scaled_transposed_flux_array.transpose()
		return scaled_flux_array, normalised_flux_error_array
	# return normalised_flux_array

def descaler(scaled_flux_array, means, stds):
	transposed_flux_array = scaled_flux_array.transpose()
	rescaled_flux_array = []
	for energy_column, mean, std in zip(transposed_flux_array, means, stds):
		rescaled_flux_array.append(energy_column * std + mean)

	rescaled_flux_array = np.array(rescaled_flux_array)
	rescaled_flux_array = rescaled_flux_array.transpose()
	rescaled_flux_array = rescaled_flux_array
	return rescaled_flux_array

y_train, scaling_means, scaling_stds, flux_errors_train = scale_flux(flux_train, flux_error_array=flux_train_error, train_mode=True)

XS_val = []
flux_val = []
flux_val_error = []
print('Fetching val data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, val_file) for val_file in val_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_val, flux_value_val, flux_error_val = future.result()
		XS_val.append(xs_values_val)
		flux_val.append(flux_value_val)
		flux_val_error.append(flux_error_val)

XS_val = np.array(XS_val)

y_val, flux_errors_val = scale_flux(flux_val, flux_error_array=flux_val_error, train_mode=False, means=scaling_means, stds=scaling_stds)
y_val = np.array(y_val)

flux_errors_val = np.array(flux_errors_val)

if test_data_directory != 'x':
	print('Fetching test data...')
	test_files = os.listdir(test_data_directory)

	XS_test = []
	flux_test = []
	flux_test_error = []
	with ProcessPoolExecutor(max_workers=data_processes) as executor:
		futures_test = [executor.submit(fetch_data, test_file) for test_file in test_files]

		for future_test in tqdm.tqdm(as_completed(futures_test), total=len(futures_test)):
			xs_values_test, flux_value_test, flux_test_err = future_test.result()
			XS_test.append(xs_values_test)
			flux_test.append(flux_value_test)
			flux_test_error.append(flux_test_err)

	XS_test = np.array(XS_test)
	y_test, flux_errors_test = scale_flux(flux_test, flux_error_array=flux_test_error, train_mode=False, means=scaling_means, stds=scaling_stds)


print('Scaling training data...')


# le_bound_index = 1 # filters out NaNs

def process_data(XS_train, XS_val, XS_test):


	channel_matrix_train = [[] for i in range(len(XS_train[0]))] # each element is a matrix of only one channel, e.g. channel_matrix[0] is all the lists containing
	channel_matrix_val = [[] for i in range(len(XS_val[0]))]
	channel_matrix_test = [[] for i in range(len(XS_test[0]))]


	# Pu-239 (n,el)
	scaled_channel_matrix_train = []
	scaled_channel_matrix_val = []
	scaled_channel_matrix_test = []


	for matrix in tqdm.tqdm(XS_train, total =len(XS_train)):
		# Each matrix has shape (num channels, points per channel)
		for channel_index, channel in enumerate(matrix):
			channel_matrix_train[channel_index].append(channel)

		# channel_matrix now has shape (num channels, num samples, points per channel)
		# Each element of channel matrix has shape (num samples, points per channel)
	for matrix in tqdm.tqdm(XS_val, total =len(XS_val)):
		for channel_index, channel in enumerate(matrix):
			channel_matrix_val[channel_index].append(channel)

	for matrix in tqdm.tqdm(XS_test, total =len(XS_test)):
		for channel_index, channel in enumerate(matrix):
			channel_matrix_test[channel_index].append(channel)

	#################################################################################################################################################################
	for channel_data_train, channel_data_val, channel_data_test in zip(channel_matrix_train, channel_matrix_val, channel_matrix_test): # each iterative variable is the tensor of one specific channel e.g. Pu-239 fission, for all samples
		transposed_matrix_train = np.transpose(channel_data_train) # shape (energy points per sample, num samples)
		transposed_matrix_val = np.transpose(channel_data_val)
		transposed_matrix_test = np.transpose(channel_data_test)

		transposed_scaled_channel_train = []
		transposed_scaled_channel_val = []
		transposed_scaled_channel_test = []
		for energy_point_train, energy_point_val, energy_point_test in zip(transposed_matrix_train[:-1], transposed_matrix_val[:-1], transposed_matrix_test[:-1]): # each point on the unionised energy grid

			train_mean = np.mean(energy_point_train)
			train_std = np.std(energy_point_train)

			scaled_point_train = zscore(energy_point_train)
			transposed_scaled_channel_train.append(scaled_point_train)

			scaled_point_val = (np.array(energy_point_val) - train_mean) / train_std
			transposed_scaled_channel_val.append(scaled_point_val)

			scaled_point_test = (np.array(energy_point_test) - train_mean) / train_std
			transposed_scaled_channel_test.append(scaled_point_test)

		scaled_channel_train = np.array(transposed_scaled_channel_train)
		scaled_channel_train = scaled_channel_train.transpose()
		scaled_channel_matrix_train.append(scaled_channel_train)

		scaled_channel_val = np.array(transposed_scaled_channel_val)
		scaled_channel_val = scaled_channel_val.transpose()
		scaled_channel_matrix_val.append(scaled_channel_val)

		scaled_channel_test = np.array(transposed_scaled_channel_test)
		scaled_channel_test = scaled_channel_test.transpose()
		scaled_channel_matrix_test.append(scaled_channel_test)

	###################################################################################################################################################################
	# print('Forming scaled training data...')
	X_matrix_train = [[] for i in range(XS_train.shape[0])] # number of samples
	X_matrix_val = [[] for i in range(XS_val.shape[0])]
	X_matrix_test = [[] for i in range(XS_test.shape[0])]

	for scaled_observable in scaled_channel_matrix_train:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_train[sample_index].append(channel_sample)

	for scaled_observable in scaled_channel_matrix_val:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_val[sample_index].append(channel_sample)

	for scaled_observable in scaled_channel_matrix_test:
		for sample_index, channel_sample in enumerate(scaled_observable):
			X_matrix_test[sample_index].append(channel_sample)

	X_matrix_train = np.array(X_matrix_train)
	X_matrix_train[np.isnan(X_matrix_train)] = 0

	X_matrix_val = np.array(X_matrix_val)
	X_matrix_val[np.isnan(X_matrix_val)] = 0 # changes nans to 0

	X_matrix_test = np.array(X_matrix_test)
	X_matrix_test[np.isnan(X_matrix_test)] = 0

	return X_matrix_train, X_matrix_val, X_matrix_test


if test_data_directory == 'x':
	XS_test = XS_val
	y_test = y_val

X_train, X_val, X_test = process_data(XS_train, XS_val, XS_test)











# Begin pytorch prep/conversion
# From here on we switch from numpy matrices to torch tensors
print('\nConverting arrays to tensors...')

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val).float()

# Working with tensors now
print('\nDefining torch classes and functions...')

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super().__init__()

		positional_encoding = torch.zeros(max_len, d_model) # matrix of zeroes with dimensions (max_len, d_model)

		position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # indices for positions

		divisor_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

		positional_encoding[:, 0::2] = torch.sin(position * divisor_term) # apply sin encoding to even positions

		positional_encoding[:, 1::2] = torch.cos(position * divisor_term) # apply cos encoding to odd positions

		positional_encoding = positional_encoding.unsqueeze(0)

		self.register_buffer('positional_encoding', positional_encoding)


	def forward(self, x):
		"""
		 x: (batch, seq_len, d_model)
		"""
		# Get current sequence length of the input
		sequence_length = x.size(1)

		# Applies positional encoding to the input embeddings
		return x + self.positional_encoding[:, :sequence_length, :]

class RegressionTransformerFeatureRows(nn.Module):
	def __init__(self,
				 num_features: int,
				 d_model: int = 64,
				 nhead: int = 4,
				 num_layers: int = 2,
				 dim_feedforward: int = 128,
				 dropout: float = 0.1,
				 max_len: int = 5000):
		super().__init__()

		self.num_features = num_features

		# model input of shape (batch, num_features, sequence_length)
		self.input_proj = nn.Linear(num_features, d_model)

		# Apply positional encoding to tokens
		self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

		# single encoder layer with
		# - multi-head self-attention
		# - feedforward network
		# - layer norm
		# - residual connections
		encoder_layer = nn.TransformerEncoderLayer( # Transformer encoder layer
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,  # ensures input tensors are (batch, seq, d_model)
		)

		# Stack multiple encoder layers into a full Transformer encoder
		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer,
			num_layers=num_layers,
		)

		# Stack multiple encoder layers into a full Transformer encoder.
		self.transformer_encoder = nn.TransformerEncoder(
			encoder_layer,
			num_layers=num_layers,
		)

		# regression head produces a single scalar (k_eff) output
		# after encoding, the sequence is pooled into a single vector of size d_model
		# sequential vanilla mlp defined below does the last bit of calculation to output a scalar value
		self.regression_head = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.ReLU(),
			nn.Linear(d_model, y_val.shape[1]),
		)

	def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
		"""
		x: (batch, num_features, seq_len)
		src_key_padding_mask: (batch, seq_len) with True where padding exists (optional)
		"""

		model_device = self.input_proj.weight.device
		# print('Model device:', model_device)
		x = x.to(model_device)

		# re orient input to make columns/tokens
		#
		# rows = features and columns = tokens, so we transpose
		#
		#   Before: (batch, num_features, seq_len)
		#   After:  (batch, seq_len, num_features)
		#
		x = x.transpose(1, 2)

		# linear projection: embed tokens into d_model-dimensional space
		# Every token now becomes a d_model-dimensional embedding
		x = self.input_proj(x)  # (batch, seq_len, d_model)

		# add positional encoding
		x = self.pos_encoder(x)

		# Transformer encoding including
		x = self.transformer_encoder(
			x,
			src_key_padding_mask=src_key_padding_mask # ignore padded tokens
		)  # (batch, seq_len, d_model)

		if src_key_padding_mask is not None: # pool sequence into single vector for regression head
			# mask-aware mean pooling (if padding is used) or simple mean pooling (no padding)
			# Mask: True = padded token, False = real token
			mask = ~src_key_padding_mask  # invert: True = valid token
			mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)

			# Zero out padded token embeddings
			x = x * mask

			# Sum over valid tokens only
			summed = x.sum(dim=1)  # (batch, d_model)
			counts = mask.sum(dim=1).clamp(min=1)
			pooled = summed / counts
		else:
			# Simple average over all tokens
			pooled = x.mean(dim=1)  # (batch, d_model)

		# regress to single scalar keff value
		# mlp regression head below
		out = self.regression_head(pooled)  # (batch, 1)

		# Return shape (batch,) instead of (batch, 1)
		return out.squeeze(-1)


class EarlyStopping:
	def __init__(self, patience, min_delta=0.0, mode="min", restore_best_weights=True):
		assert mode in ("min", "max")
		self.patience = patience
		self.min_delta = min_delta
		self.mode = mode
		self.restore_best_weights = restore_best_weights

		self.best_score = None
		self.best_state = None
		self.bad_epochs = 0
		self.should_stop = False

	def _improved(self, score, best):
		if self.mode == "min":
			return score < (best - self.min_delta)
		else:
			return score > (best + self.min_delta)

	def step(self, score, model):
		if self.best_score is None:
			self.best_score = score
			self.best_state = copy.deepcopy(model.state_dict())
			return

		if self._improved(score, self.best_score):
			self.best_score = score
			self.best_state = copy.deepcopy(model.state_dict())
			self.bad_epochs = 0
		else:
			self.bad_epochs += 1
			if self.bad_epochs >= self.patience:
				self.should_stop = True

	def restore(self, model):
		if self.restore_best_weights and self.best_state is not None:
			model.load_state_dict(self.best_state)

def iter_minibatches(X, y, batch_size, shuffle=False, device=None):
	"""
	Yields (X_batch, y_batch) with X_batch shape (B, F, T), y_batch shape (B,).
	"""
	n = X.shape[0]
	idx = torch.randperm(n) if shuffle else torch.arange(n)

	for start in range(0, n, batch_size):
		batch_idx = idx[start:start + batch_size]
		Xb = X[batch_idx]
		yb = y[batch_idx]

		if device is not None:
			Xb = Xb.to(device, non_blocking=True)
			yb = yb.to(device, non_blocking=True)

		yield Xb, yb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
N_train, F, T = X_train.shape # F stands for features i.e. number of reaction channels (T stands for tokens, each token is an energy point)
N_val = X_val.shape[0]

# define model
model = RegressionTransformerFeatureRows(num_features=F, max_len=T).to(device)

# Loss/optimisation
criterion = nn.MSELoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

print('\nDefining early stopping criteria...')

early = EarlyStopping(patience=patience, min_delta=1e-5, mode="min", restore_best_weights=True)

batch_size = 32


print('\nBeginning training...')

for epoch in tqdm.tqdm(range(1, max_epochs + 1)):
	# start training
	model.train()
	train_loss_sum = 0.0
	train_count = 0

	for Xb, yb in iter_minibatches(X_train, y_train, batch_size, shuffle=True, device=device):
		optimiser.zero_grad(set_to_none=True)
		# print("Xb device:", Xb.device)
		# print("model device:", next(model.parameters()).device)

		preds = model(Xb)              # (B,)
		loss = criterion(preds, yb)    # scalar

		loss.backward()
		optimiser.step()

		train_loss_sum += loss.item() * Xb.size(0)
		train_count += Xb.size(0)

	train_loss = train_loss_sum / max(train_count, 1)

	# --------------------
	# Validate
	# --------------------
	model.eval()
	val_loss_sum = 0.0
	val_count = 0

	with torch.no_grad():
		for Xb, yb in iter_minibatches(X_val, y_val, batch_size=64, shuffle=False, device=device):
			preds = model(Xb)
			loss = criterion(preds, yb)

			val_loss_sum += loss.item() * Xb.size(0)
			val_count += Xb.size(0)

	val_loss = val_loss_sum / max(val_count, 1)

	print(f"Epoch {epoch:03d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")

	# --------------------
	# Early stopping
	# --------------------
	early.step(val_loss, model)
	if early.should_stop:
		print(f"Early stopping. Best val MSE: {early.best_score:.6f}")
		break

# Restore best weights after training
early.restore(model)





def timed_eval(model, inputs):
	start = time.perf_counter()
	with torch.no_grad():
		out = model(inputs)
	end = time.perf_counter()
	return out, end - start


model.eval()  # switch to inference mode

with torch.no_grad():  # improve compute cost disable gradient tracking
	predictions = model(X_val.to(device))  # forward pass

out, timing = timed_eval(model, X_val)

