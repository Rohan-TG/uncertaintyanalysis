import sandy
import os
# from os.path import join
# import matplotlib.pyplot as plt
import datetime
import time

start = time.time()

za = 94239
filename = "n-094_Pu_239.endf"
tape = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
tape.to_file(filename)

number_of_samples = 5

processes = 5

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