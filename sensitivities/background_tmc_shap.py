import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import shapiq
from sklearn.linear_model import LinearRegression

feature_names = [
	"Pu9 elastic",
	"Pu9 inelastic",
	"Pu9 (n,2n)",
	"Pu9 fission",
	"Pu9 capture",

	"Pu0 elastic",
	"Pu0 inelastic",
	"Pu0 (n,2n)",
	"Pu0 fission",
	"Pu0 capture",

	"Pu1 elastic",
	"Pu1 inelastic",
	"Pu1 (n,2n)",
	"Pu1 fission",
	"Pu1 capture",
]

directory = '/home/rnt26/uncertaintyanalysis/ml/mldata/pchip-data/0-15999'
files = os.listdir(directory)

# for getting the right columns etc.
exampledf = pd.read_parquet(os.path.join(directory, files[0]))

cols = exampledf.columns
cols = cols[1:-2] # remove erg etc.

nonrealmatrix = [[] for i in range(0,15)]

keff_values = []
# main dataset loading
for f in tqdm(files, total=len(files)):
	df = pd.read_parquet(os.path.join(directory, f))
	df = df[df.ERG >= 2500]
	keff_values.append(df['keff'].values[0])

	# flat_array = []
	for ci, channel in enumerate(cols):
		nonrealmatrix[ci].append(df[channel].values)
		# flat_array += list(df[channel].values)

##############################################################################################################
# Nominal data loading and processing
keff_nominal = 0.99980
f_nominal = '/home/rnt26/uncertaintyanalysis/ml/mldata/baselines/endfbviii.0/endfbviii0_baseline_data_Pu-239_-1_Pu-240_-1_Pu-241_-1.parquet'
df_nominal = pd.read_parquet(f_nominal)
df_nominal = df_nominal[df_nominal.ERG >= 2500]

nominal_nonrealmatrix = [[] for i in range(0,15)]
for nominal_ci, nominal_channel in enumerate(cols):
	nominal_nonrealmatrix[nominal_ci].append(df_nominal[nominal_channel].values)

nominal_pcamatrix = []
for nominal_channel in nominal_nonrealmatrix:
	pca = PCA(n_components=0.999, svd_solver='full')
	X_nominal_pca = pca.fit_transform(nominal_channel)
	nominal_pcamatrix.append(X_nominal_pca)

flattened_nominal_pca_matrix = [[] for i in range(0, len(nominal_nonrealmatrix[0]))]
for nominal_pca_channel in tqdm(nominal_pcamatrix, total=len(nominal_pcamatrix)):
	for sidx, pca_nom_sample in enumerate(nominal_pca_channel):
		flattened_nominal_pca_matrix[sidx] += list(pca_nom_sample)


##############################################################################################################
# pca decomposition
pcamatrix = []
for channel in nonrealmatrix:
	pca = PCA(n_components=0.999, svd_solver='full')
	X_pca = pca.fit_transform(channel)
	pcamatrix.append(X_pca)

# Convert into the right shape (16000, n_total_modes)
flattened_pca_matrix = [[] for i in range(0, len(nonrealmatrix[0]))]
for pca_channel in tqdm(pcamatrix, total=len(pcamatrix)):
	for sample_index, pca_sample in enumerate(pca_channel):
		flattened_pca_matrix[sample_index] += list(pca_sample)

mode_number = []
for i in pcamatrix:
	mode_number.append(len(i[0]))

group_cols = {}
start = 0
groups = []
for idx, (name, mode_n) in enumerate(zip(feature_names, mode_number)):
	group_cols[name] = list(range(start, start + mode_n))
	groups.append(list(range(start, start + mode_n)))
	start += mode_n

n_groups = len(group_cols)

fpm = np.array(flattened_pca_matrix)
# Begin R^2 cooperative game
cache = {}










