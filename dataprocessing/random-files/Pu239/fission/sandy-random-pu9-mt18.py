import sandy
import os
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Pu239

lib_name = "ENDFB_80"
nucl = Pu239.ZA * 10
filename = f"{nucl}.{lib_name}"

endf6 = sandy.get_endf6_file(lib_name, 'xs', nucl)
endf6.to_file(filename)
pendf = endf6.get_pendf(err=0.0001)

num_samples = 5  # number of samples

# this generates samples for cross sections and nubar
samples = endf6.get_perturbations(
    num_samples,
    njoy_kws=dict(
        err=0.0001,
		errorr33_kws=dict(mt=[18]),
        chi=False,
        mubar=False,
        xs=True,
        nubar=False,
        verbose=True,
    ),
)



outs = endf6.apply_perturbations(
    samples,
    njoy_kws=dict(err=0.0001),   # very fast calculation, for testing
    # to_ace=True,   # produce ACE files
    to_file=True,
    # ace_kws=dict(err=0.0001, temperature=300, verbose=True, purr=True, heatr=False, thermr=False, gaspr=False),
    verbose=True,
)



ace_outs = endf6.apply_perturbations(
	samples,
	njoy_kws=dict(err=0.0001),
	to_file=True,
	verbose=True,
)


xs_0 = sandy.Xs.from_endf6(outs[0]['pendf']).data[9437]
xs_1 = sandy.Xs.from_endf6(outs[1]['pendf']).data[9437]

attempt_pendf = outs[0]['pendf']
attempt_pendf_2 = outs[1]['pendf']