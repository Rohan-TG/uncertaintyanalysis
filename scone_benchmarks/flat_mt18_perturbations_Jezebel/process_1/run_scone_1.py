import os
import numpy as np
import tqdm
import time


ACE_file_directory = '/home/rnt26/PycharmProjects/uncertaintyanalysis/data/Pu-239_MT18_p_ACE'

# perturbation_coefficients = np.arange(-0.8, 1.001, 0.001)
perturbation_coefficients = np.arange(-0.800, 0.100, 0.001)
# perturbation_coefficients = np.arange(0.100, 1.001, 0.001)


for coefficient in tqdm.tqdm(perturbation_coefficients, total=len(perturbation_coefficients)):
	libfile = 'lib1.xsfile'

	input_coefficient = round(coefficient, 3) # Coefficient string prep

	ACE_filename = f"{ACE_file_directory}/Pu-239_coeff_{input_coefficient:0.3f}_MT18.09c" # name of ACE file for this SCONE run

	with open(libfile, 'r') as file:
		lines = file.readlines()

	Pu_239_address = 511 # Line number of the Pu-239 ACE file address

	lines[Pu_239_address] = f'94239.00c; 1; {ACE_filename};\n' # Rewrite line

	with open(libfile, 'w') as file: # write to new lib1.xsfile
		file.writelines(lines)


