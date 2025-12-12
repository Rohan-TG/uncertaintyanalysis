import sandy
import sys
import os

from groupEnergies import Pu239, Reactions

computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')

lib_name = "ENDFB_80"
nucl = Pu239.ZA * 10
filename = f"{nucl}.{lib_name}"

endf6 = sandy.get_endf6_file(lib_name, 'xs', nucl)
endf6.to_file(filename)
pendf = endf6.get_pendf(err=0.0001)


num_samples = 1  # number of samples
processes = 1

fission_channel = Reactions.fission
elastic_channel = Reactions.elastic
inelastic_channel = Reactions.inelastic
capture_channel = Reactions.capture


samples = endf6.get_perturbations(
	num_samples,
	njoy_kws=dict(
		err=0.0001,
		errorr33_kws=dict(mt=[fission_channel, elastic_channel]),
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