import os
n_cores = int(input("Enter core number: "))
os.environ["OMP_NUM_THREADS"] = f"{n_cores}"
os.environ["MKL_NUM_THREADS"] = f"{n_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_cores}"
os.environ["TF_NUM_INTEROP_THREADS"] = f"{n_cores}"
os.environ["TF_NUM_INTRAOP_THREADS"] = f"{n_cores}"

import openmc

openmc.config['cross_sections'] = '/home/rnt26/PycharmProjects/uncertaintyanalysis/data/ENDFBVIII/endfb-viii.0-hdf5/cross_sections.xml'
# openmc.config['cross_sections'] = 'perturbed_cross_sections.xml'
mats = openmc.Materials()

mat = openmc.Material(1)
mat.set_density('sum')
mat.add_nuclide('Pu239', 3.7047e-02)
mat.add_nuclide('Pu240', 1.7512e-03)
mat.add_nuclide('Pu241', 1.1674e-04)
mat.add_element('Ga', 1.3752e-03)
mats.append(mat)

mats.export_to_xml()