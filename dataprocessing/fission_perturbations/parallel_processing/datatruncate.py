import pandas as pd
import os
import tqdm
import subprocess

alldata_dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/dataprocessing/fission_perturbations/parallel_processing/fulldata'

pruned_ml_dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/ml/mldata'

csv_filenames = os.listdir(alldata_dir)

for csv in tqdm.tqdm(csv_filenames, total=len(csv_filenames)):

	df = pd.read_csv(f'{alldata_dir}/{csv}')

	current_keff = df['keff'].values[0]

	if current_keff >= 0.89 and current_keff <= 1.11:
		subprocess.run(f'cp {alldata_dir}/{csv} {pruned_ml_dir}')
