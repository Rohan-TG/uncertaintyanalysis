import os
import shutil
import tqdm

Lib80x_dir = "/home/rnt26/scone/SCONE/Lib80x/Lib80x/Lib80x"

elements = os.listdir(Lib80x_dir)

endfb8_ace_dir = "/home/rnt26/PycharmProjects/uncertaintyanalysis/data/ENDFBVIII/endfb-viii.0-ace"

suffix = '.800nc'

for Element in tqdm.tqdm(elements, total=len(elements)):
	element_dir_path = os.path.join(Lib80x_dir, Element)
	if os.path.isdir(element_dir_path):
		all_elemental_ace = os.listdir(element_dir_path)
		for isotope_ace in all_elemental_ace:
			if suffix in isotope_ace:
				full_ace_path = f"{Lib80x_dir}/{Element}/{isotope_ace}"
				shutil.copy(full_ace_path, endfb8_ace_dir)

