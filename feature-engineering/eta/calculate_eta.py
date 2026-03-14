import os
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
import pandas as pd
import matplotlib.pyplot as plt


# Calc flux weighted eta

flux_data_directory = input('Flux directory: ')
xs_directory = input('XS directory: ')

all_xs = os.listdir(xs_directory)


def retrieve_data(file):

	pu9_index = int(file.split('_')[4])
	pu0_index = int(file.split('_')[6])
	pu1_index = int(file.split('_')[8].split('.')[0])

	df = pd.read_parquet(f'{xs_directory}/{file}.parquet')
	energy_grid = df['ERG'].values

	fission_xs_pu239 = df['94239_MT18_XS'].values
	capture_xs_pu239 = df['94239_MT102_XS'].values

	fission_xs_pu240 = df['94240_MT18_XS'].values
	capture_xs_pu240 = df['94240_MT102_XS'].values

	fission_xs_pu241 = df['94241_MT18_XS'].values
	capture_xs_pu241 = df['94241_MT102_XS'].values



	pu9_mt18xs = df['94239_MT18_XS'].values.tolist()
	pu0_mt18xs = df['94240_MT18_XS'].values.tolist()
	pu1_mt18xs = df['94241_MT18_XS'].values.tolist()

	pu9_mt2xs = df['94239_MT2_XS'].values.tolist()
	pu0_mt2xs = df['94240_MT2_XS'].values.tolist()
	pu1_mt2xs = df['94241_MT2_XS'].values.tolist()

	pu9_mt4xs = df['94239_MT4_XS'].values.tolist()
	pu0_mt4xs = df['94240_MT4_XS'].values.tolist()
	pu1_mt4xs = df['94241_MT4_XS'].values.tolist()

	pu9_mt16xs = df['94239_MT16_XS'].values.tolist()
	pu0_mt16xs = df['94240_MT16_XS'].values.tolist()
	pu1_mt16xs = df['94241_MT16_XS'].values.tolist()

	pu9_mt102xs = df['94239_MT102_XS'].values.tolist()
	pu0_mt102xs = df['94240_MT102_XS'].values.tolist()
	pu1_mt102xs = df['94241_MT102_XS'].values.tolist()

	XS_obj = [pu9_mt2xs, pu9_mt4xs, pu9_mt16xs, pu9_mt18xs, pu9_mt102xs,
				pu0_mt2xs, pu0_mt4xs, pu0_mt16xs, pu0_mt18xs, pu0_mt102xs,
				pu1_mt2xs, pu1_mt4xs, pu1_mt16xs, pu1_mt18xs, pu1_mt102xs,]


	flux_file = f'Flux_data_Pu-239_{pu9_index}_Pu-240_{pu0_index}_Pu-241_{pu1_index}.parquet'
	flux_read_obj = pd.read_parquet(f'{flux_data_directory}/{flux_file}', engine='pyarrow')
	flux_data = flux_read_obj['flux'].values
	flux_error = flux_read_obj['flux_errror']

	global flux_lower_bounds, flux_upper_bounds
	flux_lower_bounds = flux_read_obj['low_erg_bounds'].values
	flux_upper_bounds = flux_read_obj['high_erg_bounds'].values



