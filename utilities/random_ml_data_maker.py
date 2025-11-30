import os
processes = int(input("Enter n. processes: "))
computer = os.uname().nodename
import sys
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')

import pandas as pd
import ENDF6

isotope = input("Enter Element-nucleon_number: ")
MT = int(input("Enter MT number: "))
outputs_directory = input("Enter SCONE output directory: ")
pendf_dir = input("Enter PENDF directory: ")
parquet_directory = os.getcwd()