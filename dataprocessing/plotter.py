import matplotlib.pyplot as plt
# import sandy
import numpy as np
import endf


table = endf.ace.get_table('tmux_Pu-239_coeff_0.2_MT18.09c')

Pu239_02 = table.interpret()


fission_data = Pu239_02[18]

fission_xs = fission_data.xs