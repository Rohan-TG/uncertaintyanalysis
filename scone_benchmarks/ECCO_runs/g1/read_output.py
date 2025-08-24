import os
import pandas
import tqdm

dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/scone_benchmarks/ECCO_runs/g1/outputfiles'

files = os.listdir(dir)

keff_list = []
keff_error_list = []
coeff_list = []

for filename in tqdm.tqdm(files, total=len(files)):
	obj = open(f'{dir}/{filename}')
	if len(filename) == 14:
		coefficient = float(filename[7:12])
		coeff_list.append(coefficient)
	elif len(filename) == 15:
		coefficient = float(filename[7:13])
		coeff_list.append(coefficient)

	lines = obj.readlines()
	keffline = lines[12]

	keff_value_string = keffline[15:26]
	keff_value_float = float(keff_value_string.replace('E', 'e'))
	keff_list.append(keff_value_float)

	keff_error = keffline[27:38]
	keff_error_float = float(keff_error.replace('E', 'e'))
	keff_error_list.append(keff_error_float)