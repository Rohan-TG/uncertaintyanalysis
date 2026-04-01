import os
import sys
import ENDF6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from groupEnergies import Reactions

pendf = open('/Users/rntg/PycharmProjects/uncertaintyanalysis/ml/mldata/baseline/endfbviii0/94239_-1.pendf', 'r')

lines = pendf.readlines()

nu_section = ENDF6.find_section(lines, MF=1, MT=Reactions.nubar)
nu_energy, nu_values = ENDF6.read_table(nu_section)

# nu_prompt_section = ENDF6.find_section(lines, MF=1, MT=Reactions.nu_prompt)

