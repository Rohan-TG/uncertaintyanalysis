import sandy
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import endf

za = 94239

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
filename = '94239_master_file.ENDFB8_0'
endf6.to_file(filename)

num_samples = 5  # number of samples
processes = 5

import time
import datetime

start= time.time()

mf3_and_mf5_samples = endf6.get_perturbations(
	num_samples,
	njoy_kws=dict(
		err=0.0001,
		chi=True,
		mubar=False,
		xs=True,
		nubar=False,
		verbose=True,
		to_file=True,
	),
)


mf3_and_mf5_pendf_outs = endf6.apply_perturbations( # generates the PENDFs only
	mf3_and_mf5_samples,
	processes=processes,
	njoy_kws=dict(err=0.0001),  # low error
	to_file=True,
	verbose=True,
)

mf3_and_mf5_ace_outs = endf6.apply_perturbations(
	mf3_and_mf5_samples,
	processes=processes,
	njoy_kws=dict(err=0.0001),
	to_ace=True,
	ace_kws=dict(err=0.0001, temperature=300, verbose=True, purr=True, heatr=False, thermr=False, gaspr=False),
	verbose=True,
	to_file=True,
	)


end = time.time()
print(f'Completed in {datetime.timedelta(seconds=(end - start))}')
