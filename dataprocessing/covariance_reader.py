import sandy
from groupEnergies import Pu239
za = 94239
filename = "n-094_Pu_239.endf"
tape = sandy.Endf6.from_file(filename)


mat = Pu239.MAT
section = tape.read_section(mat, 33, 18)

out = tape.get_errorr(
    temperature=0,
    xs=True,
    nubar=False,
    chi=False,
    mubar=False,
    errorr33_kws={"mt": [18], "relative": True}
)

