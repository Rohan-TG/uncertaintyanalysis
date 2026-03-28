import sandy
import subprocess
import numpy as np
import datetime
import time
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')
from groupEnergies import Groups, Pu239, Pu240, Pu241, Ga69, Ga71, Reactions

za = Ga71.ZA


# endf6 = sandy.Endf6.from_file('/home/rnt26/PycharmProjects/uncertaintyanalysis/n-094_Pu_239.endf')
endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)



lib_name = "ENDFB_80"
nucl = Ga71.ZA * 10
filename = f"{nucl}.{lib_name}"

# endf6.to_file(filename)
pendf = endf6.get_pendf(err=0.0001)


outs = endf6.get_ace(temperature=300,
					 heatr=False,
					 thermr=False,
					 gaspr=False,
					 purr=True,
					 verbose=True,
					 pendf=pendf)

savefilename = f"Ga71_ENDFBVIII.0.03c"
with open(f"{savefilename}", mode="w") as f:
	f.write(outs["ace"])







