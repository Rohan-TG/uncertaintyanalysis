import os
import pandas

dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/scone_benchmarks/flatp_mt18_Jezebel/p_1/outputfiles'

files = os.listdir(dir)

for filename in files:
	obj = open(f'{dir}/{filename}')
	if len(filename) == 14:
		coefficient = float(filename[7:12])
	elif len(filename) == 15:
		coefficient = float(filename[7:13])

	lines = obj.readlines()
	keffline = lines[12]
	break