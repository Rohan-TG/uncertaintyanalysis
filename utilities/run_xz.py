import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import subprocess
import time
import datetime

start = time.time()

target_directory = input("Enter directory: ")
filetype = input("Enter file type: ")
processes = int(input("Enter n. processes: "))
all_files = os.listdir(target_directory)

filenames = []
for x in all_files:
	if filetype in x:
		filenames.append(x)

def run_xz(file):
	"""Runs xz on one file"""
	subprocess.run(f'xz {target_directory}/{file}', shell=True)

with ProcessPoolExecutor(max_workers=processes) as executor:
	futures = [executor.submit(run_xz, f) for f in filenames]

	for i in tqdm.tqdm(as_completed(futures), total=len(futures)):
		pass

end = time.time()

elapsed = end - start
print(f"Time elapsed: {datetime.timedelta(seconds=elapsed)}")