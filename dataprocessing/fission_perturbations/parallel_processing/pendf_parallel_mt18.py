import sandy
import numpy as np
import datetime
import time
import tqdm
# from dataprocessing import ENDF6


start = time.time()

za = 94239

# perturbation_coefficients = np.arange(-0.8, 1.001, 0.001)

# perturbation_coefficients = np.arange(-0.8, -0.6, 0.001)
# perturbation_coefficients = np.arange(-0.599, -0.4, 0.001)
# perturbation_coefficients = np.arange(-0.4, -0.2, 0.001)
# perturbation_coefficients = np.arange(-0.2, 0.0, 0.001)
# perturbation_coefficients = np.arange(0.0, 0.3, 0.001)
# perturbation_coefficients = np.arange(0.3, 0.6, 0.001)
perturbation_coefficients = np.arange(0.6, 1.001, 0.001)

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendf = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)

xs = sandy.Xs.from_endf6(pendf)

lower_bound = 1e-5  # eV
upper_bound = 2e7   # eV
domain = [lower_bound, upper_bound]


mat = 9437
mt = 18

for coeff in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):

    perturbation = sandy.Pert([1, 1 + coeff], index=domain)

    xspert = xs.custom_perturbation(mat, mt, perturbation)

    pendf_pert = xspert.to_endf6(pendf) # Create PENDF of perturbed data

    tag = "_pert"

    savefilename = f"Pu-239_coeff_{coeff:0.3f}_MT18.pendf"

    pendf_pert.to_file(savefilename)




end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")
