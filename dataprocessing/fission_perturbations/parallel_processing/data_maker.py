import pandas as pd
import os
import tqdm

xscsv_dir = '/home/rnt26/PycharmProjects/uncertaintyanalysis/dataprocessing/fission_perturbations/parallel_processing/csv_data'

keff_data = pd.read_csv('raw_keff_data.csv')

xscsv_filenames = os.listdir(xscsv_dir)




for xsfile in tqdm.tqdm(xscsv_filenames, total=len(xscsv_filenames)):
	df = pd.read_csv(f'{xscsv_dir}/{xsfile}')
	coeff = df['P'].values[0]

	xs = df['XS'].values
	erg = df['ERG'].values


	keff_p_values = keff_data['p'].values
	if coeff in keff_p_values:

		reduced_keff_df = keff_data[keff_data.p == coeff]

		# print(reduced_keff_df)
		# print(coeff)

		keff_value = reduced_keff_df['k_eff'].values[0]
		keff_error = reduced_keff_df['k_eff_err'].values[0]

		kefflist = [keff_value for i in erg]
		keff_error_list = [keff_error for i in erg]

		new_df = pd.DataFrame({'ERG': erg,
							   'XS': xs,
							   'p': coeff,
							   'keff': kefflist,
							   'keff_err': keff_error_list})

		new_df.to_csv(f'fulldata/Pu239_p{coeff}_MT18_flat.csv')