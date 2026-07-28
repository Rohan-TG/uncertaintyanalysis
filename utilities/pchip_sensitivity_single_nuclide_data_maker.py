import os
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
from groupEnergies import Reactions, pchip_energies
import tqdm
import ENDF6
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import PchipInterpolator

processes = int(input("Num. processes: "))
outputs_directory = input("Enter SCONE outputs directory: ")
output_files = os.listdir(outputs_directory)


