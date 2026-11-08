import sandy
from groupEnergies import Pu239
za = 94239
filename = "n-094_Pu_239.endf"
tape = sandy.get_endf6_file("ENDFB_80", "xs", za * 10)
# tape.to_file(filename)


mat = Pu239.MAT
section = tape.read_section(mat, 33, 18)

