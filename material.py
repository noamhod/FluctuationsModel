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
