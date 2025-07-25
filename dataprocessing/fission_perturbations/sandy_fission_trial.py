import os
n_cores = 20
os.environ["OMP_NUM_THREADS"] = f"{n_cores}"
os.environ["MKL_NUM_THREADS"] = f"{n_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_cores}"
os.environ["TF_NUM_INTEROP_THREADS"] = f"{n_cores}"
os.environ["TF_NUM_INTRAOP_THREADS"] = f"{n_cores}"

import sandy
# from os.path import join
# import matplotlib.pyplot as plt
import datetime
import time

start = time.time()

za = 94239
filename = "n-094_Pu_239.endf"

tape = sandy.get_endf6_file("ENDFB_80", "xs", za * 10).get_pendf(temperature = 300, purr=True, heatr=False, gaspr=False, verbose=True)
xs = sandy.Xs.from_endf6(tape)


mat = 9437
mt = 18
perturbation_coefficient = 0

perturbed_xs = []
for i in range(1, egrid.size):
    e_start = egrid[i-1]
    e_stop = egrid[i]
    index = egrid[i-1: i+1]
    pert = sandy.Pert([1, 1 + pert_coeff], index=index)
    print(f"perturbed xs in energy bin #{i} [{e_start:.5e}, {e_stop:.5e}]")
    xspert = xs.custom_perturbation(mat, mt, pert)
    perturbed_xs.append(xspert)

tape.to_file(filename)

number_of_samples = 1

processes = 1

cli = f"{filename}  --processes {processes}  --samples {number_of_samples}  --mf 33  --temperatures 300  --acer  --debug"
sandy.sampling.run(cli.split())



smps = tape.get_perturbations(
    number_of_samples,
    njoy_kws=dict(
        err=10,   # very fast calculation, for testing
        chi=False,
        mubar=False,
        xs=True,
        nubar=False,
        verbose=True,
    ),
)


outs = tape.apply_perturbations(
    smps,
    processes=processes,
    njoy_kws=dict(err=1),   # very fast calculation, for testing
    to_ace=True,   # produce ACE files
    to_file=True,
    ace_kws=dict(err=1, temperature=300, verbose=True, purr=False, heatr=False, thermr=False, gaspr=False),
    verbose=True,
)

end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")