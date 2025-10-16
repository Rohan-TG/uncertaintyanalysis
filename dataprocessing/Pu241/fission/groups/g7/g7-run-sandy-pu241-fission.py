import sandy
import subprocess
import numpy as np
import datetime
import time
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')
from groupEnergies import Groups, Pu241, Reactions

processes = int(input("Num. processes: "))

start = time.time()

za = Pu241.ZA
group = 7

perturbation_coefficients = np.arange(-0.500, 0.501, 0.001)

# endf6 = sandy.Endf6.from_file('/home/rnt26/PycharmProjects/uncertaintyanalysis/n-094_Pu_239.endf')
endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs = sandy.Xs.from_endf6(pendf)
heated_xs = sandy.Xs.from_endf6(pendfheated)

lower_bound = Groups.g7
upper_bound = Groups.g6
domain = [lower_bound, upper_bound]


mat = Pu241.MAT
mt = Reactions.fission


# for coeff in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):


def run_sandy(coeff):
	perturbation = sandy.Pert([1, 1 + coeff], index=domain)

	xspert = xs.custom_perturbation(mat, mt, perturbation)
	heated_xspert = heated_xs.custom_perturbation(mat, mt, perturbation)

	pendf_pert = xspert.to_endf6(pendf) # Create PENDF of perturbed data
	heated_pendf_pert = heated_xspert.to_endf6(pendfheated)

	outs = endf6.get_ace(temperature=300,
						 heatr=False,
						 thermr=False,
						 gaspr=False,
						 purr=True,
						 verbose=True,
						 pendf=pendf_pert)

	savefilename = f"Pu241_g{group}_{coeff:0.3f}_MT{mt}.09c"
	with open(f"{savefilename}", mode="w") as f:
		f.write(outs["ace"])

	savefilependf = f"Pu241_g{group}_{coeff:0.3f}_MT{mt}.pendf"
	heated_pendf_pert.to_file(savefilependf)





with ProcessPoolExecutor(max_workers = processes) as executor:
	futures = [executor.submit(run_sandy, c) for c in perturbation_coefficients]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass

end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")





