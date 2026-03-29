import sandy
import sys
import os


computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Pu239, Pu240, Pu241, Ga69, Ga71


# lib_name = "JEFF_4"
nucl = Ga71.ZA * 10
filename = f"n_31-Ga-071g.jeff"

endf6 = sandy.Endf6.from_file(filename)

pendf = endf6.get_pendf(err=0.0001)
pendf.to_file('31071_-1.pendf')

outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=pendf)
with open(f"31071_-1.03c", mode="w") as f:
	f.write(outs["ace"])
