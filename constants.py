import math
import units as U

pi       = 3.14159265358979323846
twopi    = 2*pi
re       = 2.817940326213e-15*U.m2cm # classical electron radius, cm
me       = 0.5109989500015*U.MeV2eV  # electron mass eV
mu       = 105.658*U.MeV2eV          # muon mass eV
mp       = 938.2720881629*U.MeV2eV   # proton mass eV
K        = 0.307075*U.MeV2eV         # MeV mol−1 cm2 # 4pi N_A r_e^2 m_e c^2
j        = 0.200                     # from the PDG implementation
Avogadro = 6.02214179e+23            # 1/mole
twoln10  = 2.*math.log(10.)