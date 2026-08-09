import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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