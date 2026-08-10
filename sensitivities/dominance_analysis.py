import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

feature_names = [
	"Pu9 Elastic",
	"Pu9 Inelastic",
	"Pu9 (n,2n)",
	"Pu9 fission",
	"Pu9 capture",

	"Pu0 Elastic",
	"Pu0 Inelastic",
	"Pu0 (n,2n)",
	"Pu0 fission",
	"Pu0 capture",

	"Pu1 Elastic",
	"Pu1 Inelastic",
	"Pu1 (n,2n)",
	"Pu1 fission",
	"Pu1 capture",
]


# analysis_df = df[feature_cols + ["keff"]]

directory = '/home/rnt26/uncertaintyanalysis/ml/mldata/pchip-data/0-15999'
files = os.listdir(directory)
exampledf = pd.read_parquet(os.path.join(directory, files[0]))




cols = exampledf.columns
cols = cols[1:-2]
nonrealmatrix = [[] for i in range(0,15)]

for f in tqdm(files, total=len(files)):
	df = pd.read_parquet(os.path.join(directory, f))
	df = df[df.ERG >= 2500]
	for ci, channel in enumerate(cols):
		nonrealmatrix[ci].append(df[channel].values)

pcamatrix = []
for channel in nonrealmatrix:
	pca = PCA(n_components=0.999, svd_solver='full')
	X_pca = pca.fit_transform(channel)
	pcamatrix.append(X_pca)