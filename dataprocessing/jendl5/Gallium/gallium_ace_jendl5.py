import sandy
import sys
import os


computer = os.uname().nodename
if computer == 'fermiac':
	sys.path.append('/home/rnt26/PycharmProjects/uncertaintyanalysis/')
elif computer == 'oppie':
	sys.path.append('/home/rnt26/uncertaintyanalysis/')


endf6 = sandy.Endf6.from_file('../Pu241/Pu_241_JENDL5.endf')
# pendf = endf6.get_pendf(err=0.0001, verbose=True)
outs = endf6.get_ace(temperature=300, heatr=False, thermr=False, gaspr=False, purr=True, verbose=True, pendf=pendf)
with open(f"Pu_240_JENDL5.03c", mode="w") as f:
	f.write(outs["ace"])

# pendf.to_file(f"Pu_241_JENDL5.pendf")