import sandy
import os
import datetime
# import numpy as np
import time

custom_path =  "/home/rnt26/PycharmProjects/uncertaintyanalysis"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH," "") + ":" + custom_path


from groupEnergies import Groups

start = time.time()
za = 94239

perturbation_coefficient1 = -0.95

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendfheated = endf6.get_pendf(err=0.0001, verbose=True, temperature=300)
original_pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs_unheated = sandy.Xs.from_endf6(original_pendf)
xs_heated = sandy.Xs.from_endf6(pendfheated)

first_group_lower_bound = Groups.g1  # Group 1 eV
first_group_upper_bound = Groups.g0   # Group 0 eV
first_group_domain = [first_group_lower_bound, first_group_upper_bound]

mat = 9437
mt = 18

perturbation = sandy.Pert([1, 1 + perturbation_coefficient1], index=first_group_domain)
xspert = xs_unheated.custom_perturbation(mat, mt, perturbation)
xspert_heated = xs_heated.custom_perturbation(mat, mt, perturbation)

pendf_pert = xspert.to_endf6(original_pendf) # The perturbed file, time to perturb it again
heated_pendf_pert = xspert_heated.to_endf6(pendfheated)

# tag = "_pert"
# outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=pendf_pert)
# savefilename = f"regular_single_perturbation_ace.09c"
# with open(f"{savefilename}", mode="w") as f:
# 	f.write(outs["ace"])

# heated_pendf_pert.to_file('single-perturbation-file-3.pendf')


secondary_lower_bound = Groups.g4 # group 4
secondary_upper_bound = Groups.g3 # group 3
secondary_domain = [secondary_lower_bound, secondary_upper_bound]

secondary_coefficient = 3




###########################################################################################################################
# begin secondary perturbation
xs_2_unheated = sandy.Xs.from_endf6(pendf_pert)
xs_2_heated = sandy.Xs.from_endf6(heated_pendf_pert)

secondary_perturbation = sandy.Pert([1, 1+ secondary_coefficient], index=secondary_domain)
xspert_2_unheated = xs_2_unheated.custom_perturbation(mat, mt, secondary_perturbation)

xspert_2_heated = xs_2_heated.custom_perturbation(mat, mt, secondary_perturbation)

secondary_pendf_pert = xspert_2_unheated.to_endf6(pendf_pert)
secondary_heated_pendf_pert = xspert_2_heated.to_endf6(heated_pendf_pert)

secondary_heated_pendf_pert.to_file('simtest-trial-g1g4.pendf')

secondary_outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=secondary_pendf_pert)
savefilename2 = f"dual-perturbation-ace-g1-g4.09c"
with open(f"{savefilename2}", mode="w") as f:
	f.write(secondary_outs["ace"])




end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")