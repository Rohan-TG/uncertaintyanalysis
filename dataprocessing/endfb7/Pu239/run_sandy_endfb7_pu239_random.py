import sandy
import sys
import os


computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Pu239


lib_name = "ENDFB_71"
nucl = Pu239.ZA * 10
filename = f"{nucl}.{lib_name}"

endf6 = sandy.get_endf6_file(lib_name, 'xs', nucl)
# endf6.to_file(filename)
# pendf = endf6.get_pendf(err=0.0001)


num_samples = 5  # number of samples
processes = 5



samples = endf6.get_perturbations(
	num_samples,
	njoy_kws=dict(
		err=0.0001,
		chi=False,
		mubar=False,
		xs=True,
		nubar=False,
		verbose=True,
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


