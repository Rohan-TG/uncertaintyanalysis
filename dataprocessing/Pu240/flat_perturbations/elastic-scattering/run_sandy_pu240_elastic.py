import sandy
import time
import numpy as np
import datetime
import tqdm
import sys
sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Groups, Reactions, Pu240

start = time.time()

za = Pu240.ZA


# perturbation_coefficients = np.arange(-0.500, 0.501, 0.001)
perturbation_coefficients = [-0.500, 0.500]

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs = sandy.Xs.from_endf6(pendf)
heated_xs = sandy.Xs.from_endf6(pendfheated)

lower_bound = Groups.g33  # eV
upper_bound = Groups.g0   # eV
domain = [lower_bound, upper_bound]

MAT = Pu240.MAT
MT = Reactions.elastic

for coeff in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):

    perturbation = sandy.Pert([1, 1 + coeff], index=domain)

    xspert = xs.custom_perturbation(MAT, MT, perturbation)
    heated_xspert = heated_xs.custom_perturbation(MAT, MT, perturbation)

    pendf_pert = xspert.to_endf6(pendf) # Create PENDF of perturbed data
    heated_pendf_pert = heated_xspert.to_endf6(pendfheated)

    # tag = "_pert"
    outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=pendf_pert)

    savefilename = f"Flat_Pu240_{coeff:0.3f}_MT{MT}.09c"
    with open(f"{savefilename}", mode="w") as f:
        f.write(outs["ace"])

    # savefilependf = f"Flat_Pu241_{coeff:0.3f}_MT{MT}.pendf"
    # heated_pendf_pert.to_file(savefilependf)





end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")





