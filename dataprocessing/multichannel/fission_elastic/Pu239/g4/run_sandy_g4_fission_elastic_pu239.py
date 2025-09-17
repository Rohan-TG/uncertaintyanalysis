import sandy
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.append('/home/rnt26/uncertaintyanalysis')
from groupEnergies import Pu239, Groups, Reactions
import tqdm
import time
import datetime
# import numpy as np

processes = int(input("Number of NJOY processes: "))
start = time.time()

group = 4
za = Pu239.ZA

perturbation_domain = [-0.25 , -0.236, -0.222, -0.208, -0.194, -0.18 , -0.166, -0.152,
       -0.138, -0.124, -0.11 , -0.096, -0.082, -0.068, -0.054, -0.04 ,
       -0.026, -0.012,  0.002,  0.016,  0.03 ,  0.044,  0.058,  0.072,
        0.086,  0.1  ,  0.114,  0.128,  0.142,  0.156,  0.17 ,  0.184,
        0.198,  0.212,  0.226,  0.24 ,  0.25]

perturbation_pairs = []
for i in perturbation_domain:
	for j in perturbation_domain:
		perturbation_pairs.append([i,j])


endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
original_pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs_unheated = sandy.Xs.from_endf6(original_pendf)
xs_heated = sandy.Xs.from_endf6(pendfheated)


group_lower_bound = Groups.g4 # g4
group_upper_bound = Groups.g3 # g3

group_domain = [group_lower_bound, group_upper_bound]

mat = Pu239.MAT # Pu-239
First_MT = Reactions.fission
Second_MT = Reactions.elastic

def run_sandy(pair):
	pg1 = pair[0]
	pg2 = pair[1]

	perturbation = sandy.Pert([1, 1 + pg1], index=group_domain) # Perturbation applied to the first channel, in this case fission
	xspert = xs_unheated.custom_perturbation(mat, First_MT, perturbation) # Apply perturbation
	xspert_heated = xs_heated.custom_perturbation(mat, First_MT, perturbation)

	pendf_pert = xspert.to_endf6(original_pendf)  # The perturbed file, time to perturb it again
	heated_pendf_pert = xspert_heated.to_endf6(pendfheated)

	###########################################################################################################################
	# begin secondary perturbation, in this case elastic scattering
	xs_2_unheated = sandy.Xs.from_endf6(pendf_pert)
	xs_2_heated = sandy.Xs.from_endf6(heated_pendf_pert)

	secondary_perturbation = sandy.Pert([1, 1 + pg2], index=group_domain)
	xspert_2_unheated = xs_2_unheated.custom_perturbation(mat, Second_MT, secondary_perturbation)

	xspert_2_heated = xs_2_heated.custom_perturbation(mat, Second_MT, secondary_perturbation)

	secondary_pendf_pert = xspert_2_unheated.to_endf6(pendf_pert)
	secondary_heated_pendf_pert = xspert_2_heated.to_endf6(heated_pendf_pert)

	secondary_heated_pendf_pert.to_file(f'Pu-239_g{group}_MT{First_MT}_{pg1:0.3f}_MT{Second_MT}_{pg2:0.3f}.pendf')

	secondary_outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True,
								   pendf=secondary_pendf_pert)
	savefilename2 = f"Pu-239_g{group}_MT{First_MT}_{pg1:0.3f}_MT{Second_MT}_{pg2:0.3f}.09c"
	with open(f"{savefilename2}", mode="w") as f:
		f.write(secondary_outs["ace"])






with ProcessPoolExecutor(max_workers = processes) as executor:
	futures = [executor.submit(run_sandy, c) for c in perturbation_pairs]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass

end = time.time()
elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")