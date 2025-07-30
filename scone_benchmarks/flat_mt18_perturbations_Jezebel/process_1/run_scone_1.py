import os
# import numpy as np



libfile = 'lib1.xsfile'
with open(libfile, 'r') as file:
	lines = file.readlines()

Pu_239_address = 511
lines[Pu_239_address] = 'new pu address test\n'


with open(libfile, 'w') as file:
	file.writelines(lines)