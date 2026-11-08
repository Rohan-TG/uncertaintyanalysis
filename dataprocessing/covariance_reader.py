import sys
import os
import numpy as np
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
import sandy
from groupEnergies import Pu239
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

