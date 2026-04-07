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

