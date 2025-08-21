import pandas as pd
import ENDF6
import tqdm
import os

group = 7

pendf_path = f'/home/rnt26/uncertaintyanalysis/dataprocessing/fission_perturbations/ECCO33_perturbations/group_{group}/pendf'
pendf_names  = os.listdir(pendf_path)

MT = 102

for filename in tqdm.tqdm(pendf_names, total=len(pendf_names)):
	f = open(f'{dir}/{filename}')
	lines = f.readlines()
	section = ENDF6.find_section(lines, MF=3, MT=MT)
	erg, xs = ENDF6.read_table(section)

	name_split = filename.split('_')
	coefficient = float(name_split[2])
	coeff_list = [coefficient for i in xs]

	df_temp = pd.DataFrame({'ERG': erg, 'XS': xs, 'P':coeff_list})
	df_temp.to_csv(f'ECCO-g{group}-Pu9_{coefficient}_MT102_XS.csv')