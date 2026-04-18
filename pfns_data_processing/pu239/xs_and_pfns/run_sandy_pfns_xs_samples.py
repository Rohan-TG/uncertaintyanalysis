import sandy
import os
import numpy as np
import subprocess
import endf
import tqdm

za = 94239

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
filename = '94239_master_file.ENDFB8_0'
endf6.to_file(filename)

num_samples = 5  # number of samples
processes = 5




mf5_samples = endf6.get_perturbations(
	num_samples,
	njoy_kws=dict(
		err=0.0001,
		chi=True,
		mubar=False,
		xs=False,
		nubar=False,
		verbose=True,
		to_file=True,
	),
)


mf5_outs = endf6.apply_perturbations( # generates the PENDFs only
	mf5_samples,
	processes=processes,
	njoy_kws=dict(err=0.0001),  # low error
	to_file=True,
	verbose=True,
)


filelist = os.listdir(os.getcwd())
if 'endf6_directory' not in filelist:
	subprocess.run('mkdir endf6_directory', shell=True)

if 'pendf_directory' not in filelist:
	subprocess.run('mkdir pendf_directory', shell=True)

subprocess.run('mv *.endf6 endf6_directory', shell=True)
subprocess.run('rm *.pendf', shell=True)

perturbed_mf5_endf6_files = os.listdir(f'{os.getcwd()}/endf6_directory')



num_xs_samples = 1
# perturb MF=3
for perturbed_endf6 in tqdm.tqdm(perturbed_mf5_endf6_files, total=len(perturbed_mf5_endf6_files)):

	file_index = perturbed_endf6.split('_')[-1].split('.')[0]

	perturbed_endf6_object = sandy.Endf6.from_file(f'endf6_directory/{perturbed_endf6}')

	mf3_samples = perturbed_endf6_object.get_perturbations(
		num_xs_samples,
		njoy_kws=dict(
			err=0.0001,
			chi=False,
			mubar=False,
			xs=True,
			nubar=False,
			verbose=True,
			to_file=True,
		)
	)

	mf3_outs = perturbed_endf6_object.apply_perturbations(
		mf3_samples,
		processes=1,
		njoy_kws=dict(err=0.0001),
		to_file = True,
		verbose = True,
	)

	subprocess.run(f'mv {za}_0.pendf {za}_{file_index}.pendf', shell=True)

subprocess.run('mv *.pendf pendf_directory', shell=True)

# ace_outs = endf6.apply_perturbations(
# 	samples,
# 	processes=processes,
# 	njoy_kws=dict(err=0.0001),
# 	to_ace = True,
# 	ace_kws=dict(err=0.0001, temperature=300, verbose=True, purr=True, heatr=False, thermr=False, gaspr=False),
# 	to_file=True,
# 	verbose=True,
# )
#
