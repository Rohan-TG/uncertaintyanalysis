import sandy
import subprocess
import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import random
# from os.path import join
import datetime
import time
import tqdm

start = time.time()

za = 94239


perturbation_coefficients = np.arange(0.00, 1.001, 0.2)

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
pendf = endf6.get_pendf(err=0.001, verbose=True)

xs = sandy.Xs.from_endf6(pendf)

lower_bound = 1e-5  # eV
upper_bound = 2e7   # eV
domain = [lower_bound, upper_bound]


mat = 9437
mt = 18


for coeff in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):

    perturbation = sandy.Pert([1, 1 + coeff], index=domain)

    xspert = xs.custom_perturbation(mat, mt, perturbation)

    pendf_pert = xspert.to_endf6(pendf) # Create PENDF of perturbed data

    tag = "_pert"
    outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=pendf_pert)

    savefilename = f"tmux_Pu-239_coeff_{coeff}_MT18.09c"
    with open(f"{savefilename}", mode="w") as f:
        f.write(outs["ace"])





end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")














# def run_tmux(coeff):
#     session_name = f"job_{coeff}"
#     command = f"tmux new-session -d -s {session_name} 'python3 fission_parallel_processing.py {coeff}'"
#     try:
#         subprocess.run(command, shell=True, check=True)
#         return f"Launched {session_name}"
#     except subprocess.CalledProcessError as error:
#         return(f"Failed {session_name} with error {error}")
#
#
# max_cores = 20
# batch_size = max_cores
#
# for i in range(0, len(perturbation_coefficients), batch_size):
#     batch = perturbation_coefficients[i:i+batch_size]
#     with ThreadPoolExecutor(max_workers=max_cores) as executor:
#         futures = [executor.submit(run_tmux, c) for c in batch]
#         for future in as_completed(futures):
#             print(future.result())
#
#     print(f"Waiting for batch {i // batch_size} to complete...")
#
#     time.sleep(5)