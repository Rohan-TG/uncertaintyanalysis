import pandas as pd
import os
import tqdm

xscsv_dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/dataprocessing/fission_perturbations/parallel_processing/csv_data'

keff_data = pd.read_csv('raw_keff_data.csv')

xscsv_filenames = os.listdir(xscsv_dir)

for xsfile in tqdm.tqdm(xscsv_filenames, total=len(xscsv_filenames)):
	df = pd.read_csv(f'{xscsv_dir}/{xsfile}')
	coeff = df['P'].values[0]

	reduced_keff_df = keff_data[keff_data.p == coeff]
	print(reduced_keff_df)
	break