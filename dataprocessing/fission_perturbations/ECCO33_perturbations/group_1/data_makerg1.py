import pandas as pd
import tqdm
import ENDF6
import os

keffs = pd.read_csv('g1_Pu-239_MT18_keffs.csv')

pendf_dir = '/home/rnt26/uncertaintyanalysis/dataprocessing/fission_perturbations/ECCO33_perturbations/group_1/pendf/'

pendf_names = os.listdir(pendf_dir)
length_list = []

MT = 18

for filename in tqdm.tqdm(pendf_names, total=len(pendf_names)):
	f = open(f'{pendf_dir}/{filename}')
	lines = f.readlines()
	section = ENDF6.find_section(lines, MF=3, MT=MT)
	erg, xs = ENDF6.read_table(section)

	name_split = filename.split('_')
	coefficient = float(name_split[2])
	coeff_list = [coefficient for i in xs]

	reduced_keff_df = keffs[keffs.p == coefficient]

	break

	# df_temp = pd.DataFrame({'ERG': erg, 'XS': xs, 'P':coeff_list})


	# df_temp.to_csv(f'Pu239_{coefficient}_flat_MT18_XS.csv')