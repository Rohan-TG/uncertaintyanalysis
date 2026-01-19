import pandas as pd
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

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
