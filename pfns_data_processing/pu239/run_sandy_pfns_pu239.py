import sandy
import os
import numpy as np

za = 94239

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
filename = '942390.ENDFB8_0'
endf6.to_file(filename)

num_samples = 10  # number of samples
processes = 5


# cli = f"{filename}  --processes {processes}  --samples {num_samples}  --mf 35  --temperatures 300  --acer  --debug"
# sandy.sampling.run(cli.split())


samples = endf6.get_perturbations(
	num_samples,
	njoy_kws=dict(
		err=0.0001,
		chi=True,
		mubar=False,
		xs=False,
		nubar=False,
		verbose=True,
		to_file=True,
	),
)


outs = endf6.apply_perturbations( # generates the PENDFs only
	samples,
	processes=processes,
	njoy_kws=dict(err=0.0001),  # low error
	# to_ace=True,   # produce ACE files
	to_file=True,
	# ace_kws=dict(err=0.0001, temperature=300, verbose=True, purr=True, heatr=False, thermr=False, gaspr=False),
	verbose=True,
)


ace_outs = endf6.apply_perturbations(
	samples,
	processes=processes,
	njoy_kws=dict(err=0.0001),
	to_ace = True,
	ace_kws=dict(err=0.0001, temperature=300, verbose=True, purr=True, heatr=False, thermr=False, gaspr=False),
	to_file=True,
	verbose=True,
)

