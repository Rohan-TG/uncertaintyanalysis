import sandy
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')
from groupEnergies import Pu239, Groups, Reactions
import tqdm
import time
import datetime
import numpy as np

processes = int(input("Number of NJOY processes: "))
start = time.time()


za = Pu239.ZA


first_group_lower_bound = Groups.g5  # Group 5 eV
first_group_upper_bound = Groups.g4   # Group 4 eV
first_group_domain = [first_group_lower_bound, first_group_upper_bound]

secondary_lower_bound = Groups.g6  # group 6
secondary_upper_bound = Groups.g5  # group 5
second_group_domain = [secondary_lower_bound, secondary_upper_bound]

fission_perturbation_domain = np.linspace(-0.1, 0.1, 20)

elastic_perturbation_domain = np.linspace(-0.1,0.1,20)

perturbation_pairs = []
for i in fission_perturbation_domain:
	for j in elastic_perturbation_domain:
		perturbation_pairs.append([i,j])


endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
# endf6 = sandy.Endf6.from_file('/home/rnt26/PycharmProjects/uncertaintyanalysis/n-094_Pu_239.endf')
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
original_pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs_unheated = sandy.Xs.from_endf6(original_pendf)
xs_heated = sandy.Xs.from_endf6(pendfheated)




mat = Pu239.MAT # Pu-239
First_MT = Reactions.fission
Second_MT = Reactions.elastic

def run_sandy(pair, idx):
	pg1 = pair[0]
	pg2 = pair[1]

	perturbation = sandy.Pert([1, 1 + pg1], index=first_group_domain) # Perturbation applied to the first channel, in this case fission
	xspert = xs_unheated.custom_perturbation(mat, First_MT, perturbation) # Apply perturbation
	xspert_heated = xs_heated.custom_perturbation(mat, First_MT, perturbation)

	pendf_pert = xspert.to_endf6(original_pendf)  # The perturbed file, time to perturb it again
	heated_pendf_pert = xspert_heated.to_endf6(pendfheated)

	###########################################################################################################################
	# begin secondary perturbation, in this case elastic scattering
	xs_2_unheated = sandy.Xs.from_endf6(pendf_pert)
	xs_2_heated = sandy.Xs.from_endf6(heated_pendf_pert)

	secondary_perturbation = sandy.Pert([1, 1 + pg2], index=second_group_domain)
	xspert_2_unheated = xs_2_unheated.custom_perturbation(mat, Second_MT, secondary_perturbation)

	xspert_2_heated = xs_2_heated.custom_perturbation(mat, Second_MT, secondary_perturbation)

	secondary_pendf_pert = xspert_2_unheated.to_endf6(pendf_pert)
	secondary_heated_pendf_pert = xspert_2_heated.to_endf6(heated_pendf_pert)

	secondary_heated_pendf_pert.to_file(f'94239_{idx}.pendf')

	secondary_outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True,
								   pendf=secondary_pendf_pert)
	savefilename2 = f"94239_{idx}.09c"
	with open(f"{savefilename2}", mode="w") as f:
		f.write(secondary_outs["ace"])




with ProcessPoolExecutor(max_workers = processes) as executor:
	futures = [executor.submit(run_sandy, c,idx) for idx,c in enumerate(perturbation_pairs)]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass

end = time.time()
elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")