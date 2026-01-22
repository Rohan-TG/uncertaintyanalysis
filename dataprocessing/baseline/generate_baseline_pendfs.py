import sandy
import sys
import os


computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Pu239, Pu240, Pu241


lib_name = "ENDFB_80"
nucl = Pu240.ZA * 10
filename = f"{nucl}.{lib_name}"

endf6 = sandy.get_endf6_file(lib_name, 'xs', nucl)
# endf6.to_file(filename)
pendf = endf6.get_pendf(err=0.0001)

pendf.to_file('Pu-240_ENDFBVIII.pendf')