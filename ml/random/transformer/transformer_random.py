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
