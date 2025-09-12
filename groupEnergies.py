# ECCO group structure

class Groups:
	g0 = 1.9640330000e7
	g1 = 1.0000000000e7
	g2 = 6.0653070000e6
	g3 = 3.6787940000e6
	g4 = 2.2313020000e6
	g5 = 1.3533530000e6
	g6 = 8.2085000000e5
	g7 = 4.9787070000e5
	g8 = 3.0197380000e5
	g9 = 1.8315640000e5
	g10 = 1.1109000000e5
	g11 = 6.7379470000e4
	g12 = 4.0867710000e4
	g13 = 2.4787520000e4
	g14 = 1.5034390000e4
	g15 = 9.1188200000e3
	g16 = 5.5308440000e3
	g17 = 3.3546260000e3
	g18 = 2.0346840000e3
	g19 = 1.2340980000e3
	g20 = 7.4851830000e2
	g21 = 4.5399930000e2
	g22 = 3.0432480000e2
	g23 = 1.4862540000e2
	g24 = 9.1660880000e1
	g25 = 6.7904050000e1
	g26 = 4.0169000000e1
	g27 = 2.2603290000e1
	g28 = 1.3709590000e1
	g29 = 8.3152870000e0
	g30 = 4.0000000000e0
	g31 = 5.4000000000e-1
	g32 = 1.0000000000e-1
	g33 = 1.0000100000e-5


class Nuclide:
	def __init__(self, symbol: str, Z: int, A: int, MAT: int, ZA: int):
		self.symbol = symbol
		self.Z = Z
		self.A = A
		self.MAT = MAT
		self.ZA = ZA

	def __repr__(self):
		return f"{self.symbol}-{self.A}"

Pu239 = Nuclide("Pu",Z=94, A = 239, MAT= 9437, ZA = 94239)
Pu240 = Nuclide(symbol="Pu", Z=94, A=240, ZA=94240, MAT = 9440)
Pu241 = Nuclide(symbol="Pu", Z=94, MAT=9443, ZA=94241, A=240)
Ga69 = Nuclide(symbol="Ga", Z = 31, MAT = 3125, ZA=31069, A = 69)
Ga71 = Nuclide(symbol="Ga", Z = 31, MAT = 3131, ZA=31071, A = 71)