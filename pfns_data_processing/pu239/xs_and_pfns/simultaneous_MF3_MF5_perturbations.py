import sandy
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import endf

za = 94239

endf6 = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
filename = '94239_master_file.ENDFB8_0'
endf6.to_file(filename)

num_samples = 5  # number of samples
processes = 5


