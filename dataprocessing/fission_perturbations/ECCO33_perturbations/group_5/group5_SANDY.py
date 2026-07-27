import sandy
import subprocess
import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import random
# from os.path import join
import datetime
import time
import tqdm
import sys

sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')
start = time.time()

from groupEnergies import Pu240


perturbation_coefficients = [-0.005, 0.005]
za = Pu240.ZA

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs = sandy.Xs.from_endf6(pendf)
heated_xs = sandy.Xs.from_endf6(pendfheated)

lower_bound = 1.3533530000e6  # group 5 eV
upper_bound = 	2.2313020000e6   # group 4 eV
domain = [lower_bound, upper_bound]


mat = Pu240.MAT
mt = 18


for coeff in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):

    perturbation = sandy.Pert([1, 1 + coeff], index=domain)

    xspert = xs.custom_perturbation(mat, mt, perturbation)
    heated_xspert = heated_xs.custom_perturbation(mat, mt, perturbation)

    pendf_pert = xspert.to_endf6(pendf) # Create PENDF of perturbed data
    heated_pendf_pert = heated_xspert.to_endf6(pendfheated)

    tag = "_pert"
    outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=pendf_pert)

    savefilename = f"ECCO33-g2_Pu9_{coeff:0.3f}_MT18.03c"
    with open(f"{savefilename}", mode="w") as f:
        f.write(outs["ace"])

    savefilependf = f"ECCO33-g5_Pu9_{coeff:0.3f}_MT18.pendf"
    heated_pendf_pert.to_file(savefilependf)





end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")




