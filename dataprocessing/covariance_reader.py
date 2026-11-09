import sys
import os
import numpy as np
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
import sandy
from groupEnergies import Pu239
import pandas as pd
import matplotlib.pyplot as plt

za = 94239
filename = "n-094_Pu_239.endf"
tape = sandy.Endf6.from_file(filename)


mat = Pu239.MAT
section = tape.read_section(mat, 33, 18)

out = tape.get_errorr(
    temperature=0,
    xs=True,
    nubar=False,
    chi=False,
    mubar=False,
    errorr33_kws={"mt": [18], "relative": True}
)

cov = out["errorr33"].get_cov(mts=[18]).data # extract covariance data
C18 = cov.loc[(mat, 18), (mat, 18)] # Covariances specifically for this MT channel etc.
variance = np.diag(C18.to_numpy()) # diagonal elements are the variance

if np.any(variance < -1e-12): # check if any issues with the variances
	raise ValueError("Negative covariance diagonal")

u_percent = 100 * np.sqrt(np.maximum(variance, 0)) # calculate pct. variance

result = pd.DataFrame({"E_low_eV": [x.left for x in C18.index], "E_high_eV": [x.right for x in C18.index],
					   "uncertainty_percent": u_percent}) # save in pd df
