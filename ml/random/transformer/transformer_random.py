import pandas as pd
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt


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
import time


data_directory = input('Data directory: ')
data_processes = int(input('Num. data processors: '))

all_parquets = os.listdir(data_directory)

training_fraction = float(input('Enter training data fraction: '))
lower_energy_bound = float(input('Enter lower energy bound in eV: '))
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

keff_train_mean = np.mean(keff_train)
keff_train_std = np.std(keff_train)

y_train = zscore(keff_train)


scaling_matrix_xtrain = XS_train.transpose()

scaled_columns_xtrain = []
print('Scaling training data...')


le_bound_index = 1 # filters out NaNs


training_column_means = []
training_column_stds = []

for column in tqdm.tqdm(scaling_matrix_xtrain[le_bound_index:-1], total=len(scaling_matrix_xtrain[le_bound_index:-1])):

	column_mean = np.mean(column)
	column_std = np.std(column)
	training_column_means.append(column_mean)
	training_column_stds.append(column_std)

	scaled_column = zscore(column)
	scaled_columns_xtrain.append(scaled_column)

scaled_columns_xtrain = np.array(scaled_columns_xtrain)
X_train = scaled_columns_xtrain.transpose()


XS_test = []
keff_test = []

print('Fetching test data...')


with ProcessPoolExecutor(max_workers=data_processes) as executor:
	futures = [executor.submit(fetch_data, test_file) for test_file in test_files]

	for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
		xs_values_test, keff_value_test = future.result()
		XS_test.append(xs_values_test)
		keff_test.append(keff_value_test)

XS_test = np.array(XS_test)
y_test = (np.array(keff_test) - keff_train_mean) / keff_train_std

scaling_matrix_xtest = XS_test.transpose()

scaled_columns_xtest = []
print('Scaling test data...')
for column, c_mean, c_std in tqdm.tqdm(zip(scaling_matrix_xtest[le_bound_index:-1], training_column_means, training_column_stds), total=len(scaling_matrix_xtest[le_bound_index:-1])):
	scaled_column_test = (np.array(column) - c_mean) / c_std

	scaled_columns_xtest.append(scaled_column_test)

scaled_columns_xtest = np.array(scaled_columns_xtest)
X_test = scaled_columns_xtest.transpose()


test_mask = ~np.isnan(X_test).any(axis=0)
X_test = X_test[:, test_mask]

train_mask = ~np.isnan(X_train).any(axis=0)
X_train = X_train[:, train_mask]

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
			nn.Linear(d_model, 1),
		)

	def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
		"""
		x: (batch, num_features, seq_len)
		src_key_padding_mask: (batch, seq_len) with True where padding exists (optional)
		"""

		# re orient input to make columns/tokens
		#
		# rows = features and columns = tokens, so we transpose:
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

		# mask-aware mean pooling (if padding is used) or simple mean pooling (no padding)

		if src_key_padding_mask is not None: # pool sequence into single vector for regression head
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






