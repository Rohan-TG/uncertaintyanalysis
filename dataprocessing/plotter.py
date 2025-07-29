import matplotlib.pyplot as plt
# import sandy
# import numpy as np
import ENDF6


pert0 = 'plotdata_Pu-239_coeff_0.000_MT18.pendf'

pert_m05 = 'plotdata_Pu-239_coeff_-0.500_MT18.pendf'

pert03 = 'plotdata_Pu-239_coeff_0.300_MT18.pendf'


f0 = open(pert0)
lines0 = f0.readlines()
sec0 = ENDF6.find_section(lines0, MF=3, MT=18)
x0, y0 = ENDF6.read_table(sec0)

f03 = open(pert03)
lines03 = f03.readlines()
sec03 = ENDF6.find_section(lines03, MF=3, MT=18)
x03, y03 = ENDF6.read_table(sec03)

f05 = open(pert_m05)
lines05 = f05.readlines()
sec05 = ENDF6.find_section(lines05, MF=3, MT=18)
x05, y05 = ENDF6.read_table(sec05)


plt.figure()
plt.plot(x0, y0, label='ENDF/B-VIII.0')
plt.plot(x05, y05, label='-50% Perturbation')
plt.plot(x03, y03, label = '+30% Perturbation')
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.xlabel('Energy / eV')
plt.ylabel('$\sigma_{n,f}$')
plt.title('Perturbed Pu-239 (n,f) cross sections')
plt.legend()
plt.show()