import sandy

from groupEnergies import Pu239

lib_name = "ENDFB_80"
nucl = Pu239.ZA * 10
filename = f"{nucl}.{lib_name}"

endf6 = sandy.get_endf6_file(lib_name, 'xs', nucl)
endf6.to_file(filename)

num_samples = 2  # number of samples

# this generates samples for cross sections and nubar
samples = endf6.get_perturbations(
    num_samples,
    njoy_kws=dict(
        err=1,   # very fast calculation, for testing
        chi=False,
        mubar=False,
        xs=True,
        nubar=False,
        verbose=True,
    ),
)


outs = endf6.apply_perturbations(
    samples,
    njoy_kws=dict(err=1),   # very fast calculation, for testing
    to_ace=True,   # produce ACE files
    to_file=True,
    ace_kws=dict(err=1, temperature=300, verbose=True, purr=False, heatr=False, thermr=False, gaspr=False),
    verbose=True,
)