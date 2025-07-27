import sandy
# from os.path import join
# import matplotlib.pyplot as plt
import datetime
import time

start = time.time()

za = 94239
# filename = "n-094_Pu_239.endf"

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendf = endf6.get_pendf(err=0.0001, verbose=True)

xs = sandy.Xs.from_endf6(pendf)

lower_bound = 1e-5  # eV
upper_bound = 2e7   # eV
domain = [lower_bound, upper_bound]


mat = 9437
mt = 16
perturbation_coefficient = 0.5



perturbation = sandy.Pert([1, 1 + perturbation_coefficient], index=domain)

xspert = xs.custom_perturbation(mat, mt, perturbation)

pendf_pert = xspert.to_endf6(pendf) # Create PENDF of perturbed data

tag = "_pert"
outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=pendf_pert)

savefilename = "Pu239_50pctincreased_n2n_trial.09c"
with open(f"{savefilename}", mode="w") as f:
    f.write(outs["ace"])










# perturbed_xs = []
# for i in range(1, egrid.size):
#     e_start = egrid[i-1]
#     e_stop = egrid[i]
#     index = egrid[i-1: i+1]
#     pert = sandy.Pert([1, 1 + perturbation_coefficient], index=index)
#     print(f"perturbed xs in energy bin #{i} [{e_start:.5e}, {e_stop:.5e}]")
#     xspert = xs.custom_perturbation(mat, mt, pert)
#     perturbed_xs.append(xspert)

# tape.to_file(filename)

# number_of_samples = 1
#
# processes = 1
#
# cli = f"{filename}  --processes {processes}  --samples {number_of_samples}  --mf 33  --temperatures 300  --acer  --debug"
# sandy.sampling.run(cli.split())



# smps = tape.get_perturbations(
#     number_of_samples,
#     njoy_kws=dict(
#         err=10,   # very fast calculation, for testing
#         chi=False,
#         mubar=False,
#         xs=True,
#         nubar=False,
#         verbose=True,
#     ),
# )
#
#
# outs = tape.apply_perturbations(
#     smps,
#     processes=processes,
#     njoy_kws=dict(err=1),   # very fast calculation, for testing
#     to_ace=True,   # produce ACE files
#     to_file=True,
#     ace_kws=dict(err=1, temperature=300, verbose=True, purr=False, heatr=False, thermr=False, gaspr=False),
#     verbose=True,
# )

end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")