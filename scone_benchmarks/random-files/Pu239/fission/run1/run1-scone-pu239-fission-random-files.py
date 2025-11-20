import numpy as np
import tqdm
import time
import subprocess
import datetime
import os
import sys

sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis')
from groupEnergies import Pu239, Reactions

start_time = time.time()
ZA = Pu239.ZA # ZA for Pu-240

ACE_file_directory = f'/home/rnt26/PycharmProjects/uncertaintyanalysis/data/inelastic_scattering/pu239/g{group}'
scone_executable_path = '/home/rnt26/scone/SCONE/Build/scone.out'