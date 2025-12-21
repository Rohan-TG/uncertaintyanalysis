import os
import sys
computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/') # change depending on machine
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')
from groupEnergies import Pu239, Reactions
import time
import tqdm
import subprocess

starttime = time.time()

num_cores = int(input('Num. cores: '))

default_xsfile = 'endfb-viii0.xsfile'
current_dir = os.getcwd()
target_dir = 'scone_benchmarks'
idxt = current_dir.find(target_dir)
default_Jezebel = 'Jezebel'

target_path = current_dir[:(idxt + len(target_dir))]
libfilename = f'allobservables.xsfile'

subprocess.run(f"cp {target_path}/{default_xsfile} {libfilename}", shell=True)
subprocess.run(f"cp {target_path}/{default_Jezebel} Jezebel", shell=True)


search_files = os.listdir()
for file in search_files:
	if '.xsfile' in file:
		libfile = file
		break

xsfile_fullpath = os.path.abspath(libfile)
os.environ["SCONE_ACE"] = xsfile_fullpath

if 'outputfiles' not in search_files:
	subprocess.run('mkdir outputfiles', shell=True)



Pu239_ACE_file_directory = input('Enter Pu239 ACE file directory: ')
Pu239_ACE_files = os.listdir(Pu239_ACE_file_directory)

Pu240_ACE_file_directory = input('Enter Pu240 ACE file directory: ')
Pu240_ACE_files = os.listdir(Pu240_ACE_file_directory)

Pu241_ACE_file_directory = input('Enter Pu241 ACE file directory: ')
Pu241_ACE_files = os.listdir(Pu241_ACE_file_directory)



