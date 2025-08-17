import sandy
import subprocess
import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import random
# from os.path import join
import datetime
import time
import tqdm

start = time.time()

za = 94239


perturbation_coefficients = np.arange(-0.500, 0.501, 0.001)

# perturbation_coefficients = [-0.3, -0.1, 0.1, 0.2, 0.3, 0.4]

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs = sandy.Xs.from_endf6(pendf)

lower_bound = 9.1188200000e3  # group 15 eV
upper_bound = 1.5034390000e4   # group 14 eV
domain = [lower_bound, upper_bound]


mat = 9437
mt = 18


for coeff in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):

    perturbation = sandy.Pert([1, 1 + coeff], index=domain)

    xspert = xs.custom_perturbation(mat, mt, perturbation)

    pendf_pert = xspert.to_endf6(pendf) # Create PENDF of perturbed data
    heated_pendf_pert = xspert.to_endf6(pendfheated)

    tag = "_pert"
    outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=pendf_pert)

    savefilename = f"ECCO33-g15_Pu9_{coeff:0.3f}_MT18.09c"
    with open(f"{savefilename}", mode="w") as f:
        f.write(outs["ace"])

    savefilependf = f"ECCO33-g15_Pu9_{coeff:0.3f}_MT18.pendf"
    heated_pendf_pert.to_file(savefilependf)





end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")




