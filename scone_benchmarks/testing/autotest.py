import os
import tqdm

scone_executable_path = '/home/rnt26/scone/SCONE/Build/scone.out'

files = os.listdir()

ACE_files = []
for f in files:
	if '.09c' in f:
		ACE_files.append(f)

for ACE in tqdm.tqdm(ACE_files, total=len(ACE_files)):
