import sandy
import datetime
import numpy as np
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')

from groupEnergies import Groups, Pu239, Reactions

start = time.time()
za = Pu239.ZA

perturbation_domain = np.arange(-0.25, 0.27, 0.02)

perturbation_pairs = []
for i in perturbation_domain:
	for j in perturbation_domain:
		perturbation_pairs.append([i,j])



endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
original_pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs_unheated = sandy.Xs.from_endf6(original_pendf)
xs_heated = sandy.Xs.from_endf6(pendfheated)

first_group = 3
second_group = 5

first_group_lower_bound = Groups.g3  # Group 3 eV
first_group_upper_bound = Groups.g2   # Group 2 eV
first_group_domain = [first_group_lower_bound, first_group_upper_bound]

secondary_lower_bound = Groups.g5  # group 5
secondary_upper_bound = Groups.g4  # group 4
secondary_domain = [secondary_lower_bound, secondary_upper_bound]

mat = Pu239.MAT # Pu-239
mt = Reactions.fission


# for pg1 in tqdm.tqdm(perturbation_domain, total=len(perturbation_domain)):
# 	for pg2 in perturbation_domain:
#
# 		perturbation = sandy.Pert([1, 1 + pg1], index=first_group_domain)
# 		xspert = xs_unheated.custom_perturbation(mat, mt, perturbation)
# 		xspert_heated = xs_heated.custom_perturbation(mat, mt, perturbation)
#
# 		pendf_pert = xspert.to_endf6(original_pendf) # The perturbed file, time to perturb it again
# 		heated_pendf_pert = xspert_heated.to_endf6(pendfheated)
#
#
# 		###########################################################################################################################
# 		# begin secondary perturbation
# 		xs_2_unheated = sandy.Xs.from_endf6(pendf_pert)
# 		xs_2_heated = sandy.Xs.from_endf6(heated_pendf_pert)
#
# 		secondary_perturbation = sandy.Pert([1, 1+ pg2], index=secondary_domain)
# 		xspert_2_unheated = xs_2_unheated.custom_perturbation(mat, mt, secondary_perturbation)
#
# 		xspert_2_heated = xs_2_heated.custom_perturbation(mat, mt, secondary_perturbation)
#
# 		secondary_pendf_pert = xspert_2_unheated.to_endf6(pendf_pert)
# 		secondary_heated_pendf_pert = xspert_2_heated.to_endf6(heated_pendf_pert)
#
# 		secondary_heated_pendf_pert.to_file(f'Pu9_dual_g{first_group}_{pg1:0.3f}-g{second_group}_{pg2:0.3f}_MT18.pendf')
#
# 		secondary_outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=secondary_pendf_pert)
# 		savefilename2 = f"Pu9_dual_g{first_group}_{pg1:0.3f}-g{second_group}_{pg2:0.3f}_MT18.09c"
# 		with open(f"{savefilename2}", mode="w") as f:
# 			f.write(secondary_outs["ace"])
#






def run_sandy(pair):
	pg1 = pair[0]
	pg2 = pair[1]

	perturbation = sandy.Pert([1, 1 + pg1], index=first_group_domain)
	xspert = xs_unheated.custom_perturbation(mat, mt, perturbation)
	xspert_heated = xs_heated.custom_perturbation(mat, mt, perturbation)

	pendf_pert = xspert.to_endf6(original_pendf)  # The perturbed file, time to perturb it again
	heated_pendf_pert = xspert_heated.to_endf6(pendfheated)

	###########################################################################################################################
	# begin secondary perturbation
	xs_2_unheated = sandy.Xs.from_endf6(pendf_pert)
	xs_2_heated = sandy.Xs.from_endf6(heated_pendf_pert)

	secondary_perturbation = sandy.Pert([1, 1 + pg2], index=secondary_domain)
	xspert_2_unheated = xs_2_unheated.custom_perturbation(mat, mt, secondary_perturbation)

	xspert_2_heated = xs_2_heated.custom_perturbation(mat, mt, secondary_perturbation)

	secondary_pendf_pert = xspert_2_unheated.to_endf6(pendf_pert)
	secondary_heated_pendf_pert = xspert_2_heated.to_endf6(heated_pendf_pert)

	secondary_heated_pendf_pert.to_file(f'Pu9_dual_g{first_group}_{pg1:0.3f}-g{second_group}_{pg2:0.3f}_MT18.pendf')

	secondary_outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True,
								   pendf=secondary_pendf_pert)
	savefilename2 = f"Pu9_dual_g{first_group}_{pg1:0.3f}-g{second_group}_{pg2:0.3f}_MT18.09c"
	with open(f"{savefilename2}", mode="w") as f:
		f.write(secondary_outs["ace"])





processes = int(input("Number of NJOY processes: "))

with ProcessPoolExecutor(max_workers = processes) as executor:
	futures = [executor.submit(run_sandy, c) for c in perturbation_pairs]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass



end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")