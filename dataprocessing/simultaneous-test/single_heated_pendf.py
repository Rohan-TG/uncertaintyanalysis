import sandy
# import os
import datetime
# import numpy as np
import time


start = time.time()
za = 94239

perturbation_coefficient1 = 0.0

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)

xs_heated = sandy.Xs.from_endf6(pendfheated)

lower_bound = 1e-5  # eV
upper_bound = 2e7   # eV
domain = [lower_bound, upper_bound]

mat = 9437
mt = 18

perturbation = sandy.Pert([1, 1 + perturbation_coefficient1], index=domain)
xspert_heated = xs_heated.custom_perturbation(mat, mt, perturbation)

heated_pendf_pert = xspert_heated.to_endf6(pendfheated)

tag = "_pert"
outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=heated_pendf_pert)
savefilename = f"single-heating-ace-Pu239-no-perturb.09c"
with open(f"{savefilename}", mode="w") as f:
	f.write(outs["ace"])

heated_pendf_pert.to_file('single-heating-pendf-Pu239-no-perturb.pendf')




end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")