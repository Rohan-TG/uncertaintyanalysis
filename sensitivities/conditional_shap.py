import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# import shapiq
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

# directory = '/home/rnt26/uncertaintyanalysis/ml/mldata/pchip-data/0-15999'
test_directory = '/home/rnt26/uncertaintyanalysis/ml/mldata/pchip-data/16000-17999'
files = os.listdir(test_directory)

# for getting the right columns etc.
exampledf = pd.read_parquet(os.path.join(test_directory, files[0]))

cols = exampledf.columns
cols = cols[1:-2] # remove erg etc.

nonrealmatrix = [[] for i in range(0,15)]

keff_values = []
# main dataset loading
for f in tqdm(files, total=len(files)):
	df = pd.read_parquet(os.path.join(test_directory, f))
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
X_nominal = []
for nominal_ci, nominal_channel in enumerate(cols):
	nominal_nonrealmatrix[nominal_ci].append(df_nominal[nominal_channel].values)
	X_nominal += list(df_nominal[nominal_channel].values)

X_nominal = X_nominal[1:-1]
nominal_pcamatrix = []
nominal_mode_number = []

##############################################################################################################
# pca decomposition of dataset
pca_models = []

pcamatrix = []
for channel, nominal_channel in zip(nonrealmatrix, nominal_nonrealmatrix):
	pca = PCA(n_components=0.999, svd_solver='full')
	pca_models.append(pca) # save the model for each channel

	X_pca = pca.fit_transform(channel)
	X_nominal_pca = pca.transform(nominal_channel)

	pcamatrix.append(X_pca)
	nominal_pcamatrix.append(X_nominal_pca)

for i in nominal_pcamatrix:
	nominal_mode_number.append(len(i[0]))


# Convert into the right shape (16000, n_total_modes)
flattened_pca_matrix = [[] for i in range(0, len(nonrealmatrix[0]))]
flattened_nominal_pca_matrix = [[] for i in range(0, len(nominal_nonrealmatrix[0]))]


for pca_channel in tqdm(pcamatrix, total=len(pcamatrix)):
	for sample_index, pca_sample in enumerate(pca_channel):
		flattened_pca_matrix[sample_index] += list(pca_sample)

# flatten nominal pca
for npc in nominal_pcamatrix:
	for sidx, pcasnominal in enumerate(npc):
		flattened_nominal_pca_matrix[sidx] += list(pcasnominal)

mode_number = []
for i in pcamatrix:
	mode_number.append(len(i[0]))

fpm = np.array(flattened_pca_matrix)
fnpm = np.array(flattened_nominal_pca_matrix)

X_delta = fpm - fnpm
Z_delta = np.asarray(X_delta, dtype=np.float64)
group_names = feature_names

z_mean = np.mean(Z_delta, axis=0)
z_cov = np.cov(Z_delta, rowvar=False, ddof=1)
print("Latent dimension:", Z_delta.shape[1])
print("Samples:", Z_delta.shape[0])
print("Covariance condition number:", np.linalg.cond(z_cov))

########################################################################################################################
############################## attempting size fix #####################################################################
original_group_sizes = [452] * 15

mlp_group_sizes = original_group_sizes.copy()

# Global first value removed
mlp_group_sizes[0] -= 1

# Global final value removed
mlp_group_sizes[-1] -= 1

print(mlp_group_sizes)
print(sum(mlp_group_sizes))

# Expected:
# [451, 452, 452, ..., 452, 451]
# 6778


group_slices = []

start = 0

for size in mlp_group_sizes:
	group_slices.append(
		(start, start + size)
	)

	start += size

print(group_slices)

assert group_slices[-1][1] == 6778
################################## attempting fix ######################################################################
########################################################################################################################

raw_group_sizes = [np.asarray(channel).shape[1] for channel in nonrealmatrix]
print(raw_group_sizes)
print("Total raw features:", sum(raw_group_sizes))
assert sum(raw_group_sizes) == 6780
raw_group_slices = []
start = 0

for size in raw_group_sizes:
	raw_group_slices.append((start, start + size))
	start += size



group_cols = {}
start = 0
groups = []
for idx, (name, mode_n) in enumerate(zip(feature_names, mode_number)):
	group_cols[name] = list(range(start, start + mode_n))
	groups.append(list(range(start, start + mode_n)))
	start += mode_n

n_groups = len(group_cols)



############################## Begin conditional shap

Z_perturbed = np.asarray(fpm, dtype=np.float64)
assert Z_perturbed.ndim == 2

X_nominal = np.asarray(X_nominal,dtype=np.float32)
if X_nominal.ndim == 1:
	X_nominal = X_nominal[None, :]


# from sklearn.covariance import LedoitWolf
# cov_estimator = LedoitWolf().fit(Z_perturbed)
# z_mean = cov_estimator.location_
# z_cov = cov_estimator.covariance_
n_latent = Z_perturbed.shape[1]

latent_group_slices = group_cols
group_latent_indices = [np.asarray(group, dtype=int) for group in groups]
all_group_indices = np.concatenate(group_latent_indices)
# for start, stop in latent_group_slices:
# 	group_latent_indices.append(np.arange(start, stop))

def coalition_to_indices(coalition):
	"""Convert collection of reaction group indices into corresponding PCA coordinate indices."""
	if len(coalition) == 0:
		return np.array([], dtype=int)

	return np.concatenate([group_latent_indices[g] for g in coalition])

def sample_psd_gaussian(mean, cov, n_samples, rng):
	"""Sample from a Gaussian when cov may contain small
	negative eigenvalues due to float arithmetic."""

	cov = 0.5 * (cov + cov.T)

	eigenvalues, eigenvectors = np.linalg.eigh(cov)

	# Remove tiny negative numerical eigenvalues
	eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

	L = (eigenvectors * np.sqrt(eigenvalues))

	noise = rng.standard_normal(size=(n_samples, len(mean)))

	return mean + noise @ L.T



# big function

#sample conditional pca distribution

def sample_conditional_z(z_target, coalition, n_samples, rng, ridge=1e-10):
	"""Draw samples from p(Z_notS | Z_S = z_target_S)
	while holding every PCA coordinate belonging to the
	coalition exactly equal to the target realization."""

	z_target = np.asarray(z_target, dtype=np.float64)

	conditioned_idx = coalition_to_indices(coalition)

	all_idx = np.arange(n_latent)

	unconditioned_idx = np.setdiff1d(all_idx, conditioned_idx, assume_unique=False)
	# sample directly from full joint distribution

	if len(conditioned_idx) == 0:
		return sample_psd_gaussian(z_mean, z_cov, n_samples, rng)

	if len(unconditioned_idx) == 0:

		return np.repeat(z_target[None, :], n_samples, axis=0)

	# partition mean
	mu_A = z_mean[conditioned_idx]
	mu_B = z_mean[unconditioned_idx]

	#partition covariance
	Sigma_AA = z_cov[np.ix_(conditioned_idx, conditioned_idx)]

	Sigma_BB = z_cov[np.ix_(unconditioned_idx, unconditioned_idx)]

	Sigma_BA = z_cov[np.ix_(unconditioned_idx,conditioned_idx)]

	Sigma_AB = Sigma_BA.T

	# regularisation for matrix inversion
	Sigma_AA_reg = (Sigma_AA + ridge * np.eye(len(conditioned_idx)))

	# solve with np.linalg
	delta_A = (z_target[conditioned_idx] - mu_A)

	solved_delta = np.linalg.solve(Sigma_AA_reg,delta_A)

	conditional_mean_B = (mu_B+ Sigma_BA @ solved_delta)

	solved_cov = np.linalg.solve(Sigma_AA_reg,Sigma_AB)

	conditional_cov_B = (Sigma_BB- Sigma_BA @ solved_cov)

	conditional_cov_B = (0.5 * (conditional_cov_B + conditional_cov_B.T))

	# sample missing pca coords
	sampled_B = sample_psd_gaussian(conditional_mean_B, conditional_cov_B, n_samples, rng)

	# Construct complete PCA vectors

	Z_samples = np.empty((n_samples, n_latent), dtype=np.float64)

	Z_samples[:, conditioned_idx] = (z_target[conditioned_idx])

	Z_samples[:, unconditioned_idx] = (sampled_B)

	return Z_samples



# next, reconstruct original mlp input


def reconstruct_X_from_delta_Z(Z_delta):
	Z_delta = np.asarray(Z_delta, dtype=np.float64)

	if Z_delta.ndim == 1:
		Z_delta = Z_delta[None, :]

	n_samples = Z_delta.shape[0]
	# Reconstruct the full physical 6780 length vector before fix for mlp shape

	full_nominal = np.concatenate([nominal_nonrealmatrix[g][0] for g in range(15)]).astype(np.float64)
	assert full_nominal.shape == (6780,)

	X_full = np.repeat(full_nominal[None, :], n_samples, axis=0)
	raw_start = 0

	for g in range(15):
		latent_idx = np.asarray(groups[g], dtype=int)
		delta_Z_g = Z_delta[:,latent_idx]
		components = (pca_models[g].components_)

		# Reconstruct perturbation in original 452 point energy space
		delta_X_g = (delta_Z_g @ components)

		n_raw = components.shape[1]

		X_full[:, raw_start:raw_start + n_raw] += delta_X_g
		raw_start += n_raw

	assert X_full.shape[1] == 6780 # check it is in original format

	# now remove first and final to match mlp format
	X_mlp = X_full[:, 1:-1]

	assert X_mlp.shape[1] == 6778
	return X_mlp.astype(np.float32)



# Check it has worked as intended ######################################################################################
zero_delta = np.zeros((1, X_delta.shape[1]))

X_zero = reconstruct_X_from_delta_Z(zero_delta)

print("Maximum nominal reconstruction difference:", np.max(np.abs(X_zero - X_nominal)))

np.testing.assert_allclose(X_zero, X_nominal, rtol=1e-6, atol=1e-6)
####### End of check ###################################################################################################

# remember to load model explicitly if not running this code in a training script
# selected_best_models =[1] # placeholder for no error
model = selected_best_models[0]
# runs selected model predictions
def predict_mlp(X, batch_size=4096):
	X = np.asarray(X, dtype=np.float32)

	y = model.predict(X, batch_size=batch_size,	verbose=0)

	return(np.asarray(y).reshape(-1))



def make_coalition_value_function(z_target, n_conditional_samples=256,random_seed=42):
	"""Return a cached coalition value function for one realisation being explained
	Coalitions are represented by an integer bit mask"""

	z_target = np.asarray(z_target,	dtype=np.float64)
	cache = {}

	def coalition_value(mask):
		if mask in cache:
			return cache[mask]

		coalition = [g for g in range(n_groups)	if mask & (1 << g)]
		if len(coalition) == n_groups:
			X_target = reconstruct_X_from_delta_Z(z_target[None, :])

			value = float(predict_mlp(X_target)[0])
			cache[mask] = value

			return value

		seed_sequence = np.random.SeedSequence([random_seed, int(mask)])

		rng = np.random.default_rng(seed_sequence)

		Z_samples = sample_conditional_z(z_target=z_target,	coalition=coalition,
										 n_samples=n_conditional_samples,
										 rng=rng)


		X_samples = reconstruct_X_from_delta_Z(Z_samples)
		predictions = predict_mlp(X_samples)
		value = float(np.mean(predictions))
		cache[mask] = value

		return value

	return coalition_value, cache





def conditional_group_shap(z_target, n_permutations=2000, n_conditional_samples=256, random_seed=42):

	coalition_value, cache = (make_coalition_value_function(z_target=z_target,
															n_conditional_samples=n_conditional_samples,
															random_seed=random_seed,))

	permutation_rng = np.random.default_rng(random_seed + 100000)

	# Store each permutation's marginal contributions
	permutation_contributions = np.zeros((n_permutations, n_groups),	dtype=np.float64)

	for p in tqdm(range(n_permutations)):
		permutation = permutation_rng.permutation(n_groups)
		mask = 0
		previous_value = coalition_value(mask)

		for g in permutation:
			new_mask = (mask | (1 << int(g)))
			new_value = coalition_value(new_mask)

			permutation_contributions[p, g] = (new_value - previous_value)
			mask = new_mask
			previous_value = new_value

	phi = np.mean(permutation_contributions,axis=0) #shap values
	# include mc uncertainty
	phi_se = (np.std(permutation_contributions,	axis=0,	ddof=1)	/ np.sqrt(n_permutations))

	empty_mask = 0

	full_mask = ((1 << n_groups) - 1)

	baseline_value = coalition_value(empty_mask)
	target_value = coalition_value(full_mask)
	shap_sum = np.sum(phi)
	expected_difference = (target_value	- baseline_value)
	additivity_error = (shap_sum - expected_difference)

	results = {
		"phi": phi,
		"phi_se": phi_se,
		"baseline_value": baseline_value,
		"target_value": target_value,
		"shap_sum": shap_sum,
		"expected_difference": expected_difference,
		"additivity_error": additivity_error,
		"n_unique_coalitions": len(cache),
		"permutation_contributions":
			permutation_contributions,
	}

	return results



# for one perturbed sample:
sample_index = 0

z_target = Z_perturbed[sample_index]


results = conditional_group_shap(z_target=z_target,
								 n_permutations=1000,
								 n_conditional_samples=256,
								 random_seed=42)


phi_conditional = results["phi"]
phi_se = results["phi_se"]


print("\nConditional group Shapley values")
print("--------------------------------")

for name, value, se in zip(group_names,	phi_conditional, phi_se):
	print(f"{name:20s}: {value:+.6e} +/- {se:.2e}")


print("\nBaseline E[f(X)]:", results["baseline_value"])

print("Prediction f(x):", results["target_value"])

print("Sum of conditional SHAP:", results["shap_sum"])

print("f(x) - E[f(X)]:", results["expected_difference"])

print("Additivity error:", results["additivity_error"])

print("Unique coalitions evaluated:", results["n_unique_coalitions"])



indices_to_explain = np.arange(min(200, len(Z_perturbed)))

all_conditional_phi = []

for count, idx in enumerate(indices_to_explain):
	print(f"Explaining {count + 1} /{len(indices_to_explain)}")

	results = conditional_group_shap(
		z_target=Z_perturbed[idx],
		n_permutations=1000,
		n_conditional_samples=256,
		random_seed=42 + idx)

	all_conditional_phi.append(results["phi"])

all_conditional_phi = np.asarray(all_conditional_phi)

print("Conditional SHAP array:", all_conditional_phi.shape)



# plot

conditional_importance = np.mean(np.abs(all_conditional_phi), axis=0)
order = np.argsort(conditional_importance)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(np.asarray(group_names)[order], conditional_importance[order])
ax.set_xlabel("Mean absolute conditional Shapley contribution")
ax.set_ylabel("Reaction-channel group")
plt.tight_layout()
# plt.show()