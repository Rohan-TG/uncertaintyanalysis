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


# analysis_df = df[feature_cols + ["keff"]]

directory = '/home/rnt26/uncertaintyanalysis/ml/mldata/pchip-data/0-15999'
files = os.listdir(directory)
exampledf = pd.read_parquet(os.path.join(directory, files[0]))




cols = exampledf.columns
cols = cols[1:-2]
nonrealmatrix = [[] for i in range(0,15)]
flattened_matrix = []

keff_values = []

for f in tqdm(files, total=len(files)):
	df = pd.read_parquet(os.path.join(directory, f))
	df = df[df.ERG >= 2500]
	keff_values.append(df['keff'].values[0])

	flat_array = []
	for ci, channel in enumerate(cols):
		nonrealmatrix[ci].append(df[channel].values)
		flat_array += list(df[channel].values)
	flattened_matrix.append(flat_array)



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

def r2_game(coalitions):
	"""argumetn shape (n_coalitions, n_groups)"""
	values = np.zeros(len(coalitions))

	for row, coalition in enumerate(tqdm(coalitions, total=len(coalitions), desc="Evaluating Coalitions")):
		key = tuple(coalition.astype(bool))

		if key in cache:
			values[row] = cache[key]
			continue

		selected_groups = np.flatnonzero(coalition)
		# Empty coalition
		if len(selected_groups) == 0:
			values[row] = 0.0
			cache[key] = 0.0
			continue

		# All PCA columns in selected groups
		cols = np.concatenate([groups[g] for g in selected_groups])

		X_subset = fpm[:, cols]

		model = LinearRegression()
		model.fit(X_subset, keff_values)

		r2 = model.score(X_subset, keff_values)

		values[row] = r2
		cache[key] = r2

	return values

computer = shapiq.ExactComputer(
    r2_game,
    n_players=15
)

sv = computer(index="SV", order=1)


# Extract contributions
shapley_r2 = np.array([sv[(i,)] for i in range(15)])

full_mask = np.ones((1, 15), dtype=bool)
full_r2 = r2_game(full_mask)[0]

print(full_r2) # Checks how closely the sum of shapley contributions matches the actual sum of everything
variance_fractions = shapley_r2 / full_r2

results = pd.DataFrame({
    "channel": feature_names,
    "shapley_R2": shapley_r2,
    "fraction_explained": variance_fractions,
    "percent_explained": 100 * variance_fractions,
})

results = results.sort_values(
    "fraction_explained",
    ascending=False
)

print(results)



##### Much faster approximator approach:
approximator = shapiq.PermutationSamplingSV( n=15, random_state=42,)

sv = approximator.approximate(budget=3000, game=r2_game,)

shapley_r2 = np.array([ sv[(i,)] for i in range(15)])

full_r2 = r2_game(np.ones((1, 15), dtype=bool))[0]

variance_fractions = shapley_r2 / full_r2