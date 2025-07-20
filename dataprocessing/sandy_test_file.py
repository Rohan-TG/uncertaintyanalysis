import sandy
import pandas as pd
from os.path import join
import numpy as np
import random, sys
import matplotlib.pyplot as plt

za = 94239
tape = sandy.get_endf6_file("ENDFB_80", "nfpy", za * 10)