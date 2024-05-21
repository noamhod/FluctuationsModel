import array
import math
import numpy as np
import units as U
import constants as C


class Material:
    def __init__(self, name, notation,rho,Z,A,I,Tc,densities,nelm):
        self.name = name
        self.notation = notation
        self.rho = rho # g/cm3
        self.Z   = Z   # atomic number (Z) --> np.array!
        self.A   = A   # atomic mass (A), g/mole --> np.array!
        self.I   = I   # mean excitation energy (I), eV
        self.Tc  = Tc  # production threshold for delta ray
        self.ZoA = self.avgZoA()
        self.Ep  = self.Eplasma()
        self.den = densities
        self.namedden = {}
        self.SetDensities()
        self.nElements = nelm
        self.shellcorrvec = [0,0,0]
        self.SetShellCorrVec() ## must call after SetDensities()

    def __str__(self):
        return f"{self.name}({self.notation})"

    def avgZoA(self):
        return np.mean(np.divide(self.Z,self.A))

    ### definition in Table 33.1 from PDG: https://pdg.lbl.gov/2016/reviews/rpp2016-rev-passage-particles-matter.pdf
    def Eplasma(self):
        return math.sqrt(self.rho*self.ZoA)*28.816 # eV
    
    def SetDensities(self):        
        ## https://geant4.kek.jp/lxr/source/materials/src/G4IonisParamMat.cc
        ## https://geant4.kek.jp/lxr/source/materials/src/G4DensityEffectData.cc
        self.namedden.update({"PlasmaEnergy":            self.den[0]})
        self.namedden.update({"AdjustmentFactor":        self.den[1]})
        self.namedden.update({"Cdensity":                self.den[2]})
        self.namedden.update({"X0density":               self.den[3]})
        self.namedden.update({"X1density":               self.den[4]})
        self.namedden.update({"Adensity":                self.den[5]})
        self.namedden.update({"Mdensity":                self.den[6]})
        self.namedden.update({"Delta0density":           self.den[7]})
        self.namedden.update({"ErrorDensity":            self.den[8]})
        self.namedden.update({"MeanIonisationPotential": self.den[9]})
        ## atoms and electron density
        self.fMassFractionVector = 1 #TODO: this is only true for basic elements, not compounds
        self.numberOfAtomsPerVolume = C.Avogadro * self.rho * self.fMassFractionVector / self.A[0] # 1/mole * g/cm3 * 1/(g/mole) = 1/cm3
        self.electronDensity = self.numberOfAtomsPerVolume * self.Z[0] # 1/cm3
    
    def SetShellCorrVec(self):
        # rate = 0.001 * fMeanExcitationEnergy / eV
        rate = 0.001 * self.namedden["MeanIonisationPotential"] ##TODO units? 
        rate2 = rate*rate
        self.shellcorrvec[0] = (0.422377 + 3.858019 * rate) * rate2
        self.shellcorrvec[1] = (0.0304043 - 0.1667989 * rate) * rate2
        self.shellcorrvec[2] = (-0.00038106 + 0.00157955 * rate) * rate2


#############################
### some predefined mateirals

rho_Si = 2.329     # Silicon, g/cm3
Z_Si   = [14]      # Silicon atomic number (Z)
A_Si   = [28.0855] # Silicon atomic mass (A)
I_Si   = 173.0     # Silicon mean excitation energy (I), eV
Ep_Si  = 31.05     # Silicon plasma energy (E_p), eV
Tc_Si  = 990       # Silicon, production threshold for delta ray, eV
den_Si = [31.055, 2.103, 4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14, 0.059, 173.]
nel_Si = 1
Si = Material("Silicon","Si",rho_Si,Z_Si,A_Si,I_Si,Tc_Si,den_Si,nel_Si)

rho_Al = 2.699     # Aluminum, g/cm3
Z_Al   = [13]      # Aluminum atomic number (Z)
A_Al   = [26.98]   # Aluminum atomic mass (A)
I_Al   = 166.0     # Aluminum mean excitation energy (I), eV
Ep_Al  = 32.86     # Aluminum plasma energy (E_p), eV
Tc_Al  = 990       # Aluminum production threshold for delta ray, eV
den_Al = [32.86, 2.18, 4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12, 0.061, 166.]
nel_Al = 1
Al = Material("Aluminum","Al",rho_Al,Z_Al,A_Al,I_Al,Tc_Al,den_Al,nel_Al)