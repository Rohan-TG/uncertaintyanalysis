import sandy
import sys
import os
import numpy as np
import random
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Ga71, Groups
import time


num_processes = int(input('Num cores: '))
num_files = int(input('Number of files to generate: '))

start = time.time()

lib_name = "ENDFB_80"
nucl = Ga71.ZA * 10
filename = f"{nucl}.{lib_name}"

endf6 = sandy.get_endf6_file(lib_name, 'xs', nucl)


perturbation_coefficients = np.arange(-0.40, 0.41, 0.01)

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", nucl)

pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
pendf = endf6.get_pendf(err=0.0001, verbose=True)


mat = Ga71.MAT

sensitive_MTs = [2, 4, 16, 102]


domains = [[Groups.g1, Groups.g0],
		   [Groups.g2, Groups.g1],
		   [Groups.g3, Groups.g2],
		   [Groups.g4, Groups.g3],
		   [Groups.g5, Groups.g4],
		   [Groups.g6, Groups.g5],
		   [Groups.g7, Groups.g6],
		   [Groups.g8, Groups.g7],
		   [Groups.g9, Groups.g8],
		   [Groups.g10, Groups.g9],
		   [Groups.g11, Groups.g10],
		   [Groups.g12, Groups.g11],
		   [Groups.g13, Groups.g12],
		   [Groups.g14, Groups.g13],
		   [Groups.g15, Groups.g14],
		   [Groups.g16, Groups.g15],
		   [Groups.g17, Groups.g16],
		   [Groups.g33, Groups.g17],]

index_numbers = list(range(0, num_files))




def generate_gallium_files(idx):
	working_perturbed_endf6 = [pendf]
	working_perturbed_endf6_heated = [pendfheated]

	for i, energy_bounds in enumerate(domains):

		channel_1 = sensitive_MTs[0]

		group_perturbation_coefficient_1 = round(random.uniform(-0.50, 0.50),2)

		perturbation = sandy.Pert([1, 1 + group_perturbation_coefficient_1], index = energy_bounds)

		xs = sandy.Xs.from_endf6(working_perturbed_endf6[i])
		heated_xs = sandy.Xs.from_endf6(working_perturbed_endf6_heated[i])

		xspert = xs.custom_perturbation(mat, channel_1, perturbation)
		heated_xspert = heated_xs.custom_perturbation(mat, channel_1, perturbation)

		pendf_pert = xspert.to_endf6(pendf)  # Create PENDF of perturbed data
		heated_pendf_pert = heated_xspert.to_endf6(pendfheated)

		#### begin secondary perturbation, second MT

		channel_2 = sensitive_MTs[1]

		group_perturbation_coefficient_2 = round(random.uniform(-0.50, 0.50),2)

		xs_2_unheated = sandy.Xs.from_endf6(pendf_pert)
		xs_2_heated = sandy.Xs.from_endf6(heated_pendf_pert)

		secondary_perturbation = sandy.Pert([1, 1 + group_perturbation_coefficient_2], index= energy_bounds)
		xspert_2_unheated = xs_2_unheated.custom_perturbation(mat, channel_2, secondary_perturbation)

		xspert_2_heated = xs_2_heated.custom_perturbation(mat, channel_2, secondary_perturbation)

		pendf_pert_2 = xspert_2_unheated.to_endf6(pendf_pert)
		heated_pendf_pert_2 = xspert_2_heated.to_endf6(heated_pendf_pert)


		#### begin 3rd perturbation

		channel_3 = sensitive_MTs[2]

		group_perturbation_coefficient_3 = round(random.uniform(-0.50, 0.50),2)

		xs_3_unheated = sandy.Xs.from_endf6(pendf_pert_2)
		xs_3_heated = sandy.Xs.from_endf6(heated_pendf_pert_2)

		perturbations_3 = sandy.Pert([1, 1 + group_perturbation_coefficient_3], index = energy_bounds)
		xspert_3_unheated = xs_3_unheated.custom_perturbation(mat, channel_3, perturbations_3)

		xspert_3_heated = xs_3_heated.custom_perturbation(mat, channel_3, perturbations_3)

		pendf_pert_3 = xspert_3_unheated.to_endf6(pendf_pert)
		heated_pendf_pert_3 = xspert_3_heated.to_endf6(heated_pendf_pert)


		##### perturbation of 4th channel

		channel_4 = sensitive_MTs[3]

		group_perturbation_coefficient_4 = round(random.uniform(-0.50, 0.50),2)

		xs_4_unheated = sandy.Xs.from_endf6(pendf_pert_3)
		xs_4_heated = sandy.Xs.from_endf6(heated_pendf_pert_3)

		perturbations_4 = sandy.Pert([1, 1 + group_perturbation_coefficient_4], index = energy_bounds)
		xspert_4_unheated = xs_4_unheated.custom_perturbation(mat, channel_4, perturbations_4)

		xspert_4_heated = xs_4_heated.custom_perturbation(mat, channel_4, perturbations_4)

		pendf_pert_4 = xspert_4_unheated.to_endf6(pendf_pert)
		heated_pendf_pert_4 = xspert_4_heated.to_endf6(heated_pendf_pert)

		outs = endf6.get_ace(temperature=300,
							 heatr=False,
							 thermr=False,
							 gaspr=False,
							 purr=True,
							 verbose=True,
							 pendf=pendf_pert_4)

		savefile_ace_name = f"3171_{idx}.03c"
		with open(f"{savefile_ace_name}", mode="w") as f:
			f.write(outs["ace"])



		working_perturbed_endf6.append(pendf_pert_4)
		working_perturbed_endf6_heated.append(heated_pendf_pert_4)


	heated_pendf_pert_4.to_file(f'3171_{idx}.pendf')


with ProcessPoolExecutor(max_workers=num_processes) as executor:
	futures = [executor.submit(generate_gallium_files, file_index) for file_index in index_numbers]

	for j in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass

end = time.time()
import datetime
elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")
