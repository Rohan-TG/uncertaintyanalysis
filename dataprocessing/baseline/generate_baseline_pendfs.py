import sandy
import sys
import os


computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Pu239, Pu240, Pu241, Ga69, Ga71


lib_name = "ENDFB_81"
nucl = Ga69.ZA * 10
filename = f"{nucl}.{lib_name}"

# endf6 = sandy.get_endf6_file(lib_name, 'xs', nucl)
endf_name = 'n-031_Ga_069.endf'
endf6 = sandy.Endf6.from_file(f'/home/rnt26/uncertaintyanalysis/dataprocessing/baseline/endfbviii.1/neutrons-version.VIII.1/{endf_name}')
# endf6.to_file(filename)
pendf = endf6.get_pendf(err=0.0001)

pendf.to_file('Ga-69_ENDFBVIII_1.pendf')



outs = endf6.get_ace(temperature=300,
					 heatr=False,
					 thermr=False,
					 gaspr=False,
					 purr=True,
					 verbose=True,
					 pendf=pendf)

savefilename = f"Ga-69_ENDFBVIII_1.03c"
with open(f"{savefilename}", mode="w") as f:
	f.write(outs["ace"])