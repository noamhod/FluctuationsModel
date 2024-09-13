import array
import math
import numpy as np
import FluctuationsModel.units as U
import FluctuationsModel.constants as C


#####################################################################
#####################################################################
#####################################################################


class Particle:
    def __init__(self,name,meV,mamu,chrg,spin,lepn,magm):
        self.name    = name
        self.meV     = meV
        self.mamu    = mamu
        self.spin    = spin
        self.chrg    = chrg # charge
        self.lepn    = lepn # lepton number 
        ## G4double mN = eplus*hbar_Planck/2./(proton_mass_c2 /c_squared);
        self.magm    = magm * (C.hbar_Planck/2./(C.mp/C.c2)) # pdg magnetic moment * [(eV*s) / (eV / (cm^2/s^2))] --> [s / (s^2 / cm^2))] --> [cm^2/s]
        self.isAlpha = (self.name=="alpha")
        self.isIon   = (not self.isAlpha and self.chrg>1.1)
        self.chrg2   = self.chrg**2
        self.mratio  = C.me/self.meV
        self.aMag    = 1./(0.5*C.hbar_Planck*C.c2) ## [1/((eV s)*(cm^2/s^2))] = [s/(eV * cm^2)]
        self.magm1   = self.magm*self.meV*self.aMag ## [cm^2/(s eV)] * [eV] * [s/cm^2] = []
        self.magm2   = self.magm1**2 - 1. ## []
        self.mamu27  = pow(self.mamu,0.27) if(self.meV*U.eV2GeV>1.) else -1
        self.ffact   = self.formfact()
    
    def formfact(self):
        ffact = 0.0
        if(self.lepn==0):
            x = 0.8426*U.MeV2eV
            if(self.spin==0. and self.meV*U.eV2GeV<1):
                x = 0.736*U.MeV2eV
            elif(self.meV*U.eV2GeV>1.):
                iz = int(abs(self.chrg))
                if(iz>1): x /= self.mamu27
            ffact = 2.*C.me/(x**2)
        return ffact
        


proton = Particle(name="proton",meV=938.27208816*U.MeV2eV,mamu=1.007276466621,chrg=+1.,spin=0.5,lepn=0,magm=2.79284734463)
# print(f"name={prt.name}")
# print(f"meV={prt.meV}")
# print(f"mamu={prt.mamu}")
# print(f"spin={prt.spin}")
# print(f"chrg={prt.chrg}")
# print(f"lepn={prt.lepn}")
# print(f"magm={prt.magm}")
# print(f"isAlpha={prt.isAlpha}")
# print(f"isIon={prt.isIon}")
# print(f"chrg2={prt.chrg2}")
# print(f"ratio={prt.mratio}")
# print(f"aMag={prt.aMag}")
# print(f"magm1={prt.magm1}")
# print(f"magm2={prt.magm2}")
# print(f"mamu27={prt.mamu27}")
# print(f"ffact={prt.ffact}")