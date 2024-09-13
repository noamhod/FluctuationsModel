import array
import math
import numpy as np
import ROOT
import units as U
import FluctuationsModel.constants as C
import FluctuationsModel.material as mat
import FluctuationsModel.particle as prt
import tables as tab
import bins


#####################################################################
#####################################################################
#####################################################################


class Parameters:
    def __init__(self,primprt,material,dedxmodel,dEdx_table_file):
        self.m       = primprt.meV
        self.z       = primprt.chrg
        self.spin    = primprt.spin
        self.primprt = primprt
        self.mat     = material
        self.dedxmod = dedxmodel
        # if(self.dedxmod!="BB:Tcut" and self.dedxmod!="G4:Tcut"):
        #     print(f"Unknown model named {self.dedxmod}. Quitting")
        #     quit()
        assert self.dedxmod in ["BB:Tcut","BB:Tmax","G4:Tcut"], f"Unknown model named {self.dedxmod}. Quitting"
        # print(f"Using dE/dx model: {self.dedxmod}")
        self.tbl     = tab.Tables(dEdx_table_file)
        
        ### default Energy Loss Fluctuations model used in main Physics List:
        self.r        = 0.56 #0.55 ## rate (fraction) of the ionization part of the total loss (excitation+ionization)
        self.kap      = 10 ## lower limit of the number of interactions of the particle in a step
        self.minloss  = 10. ## eV
        self.NminBohr = 10
        self.avZ      = np.mean(self.mat.Z)
        self.f2       = 0.
        # self.f2       = 0. if(self.avZ==1) else 2./self.avZ
        self.f1       = 1.-self.f2
        self.E2       = 10.*(self.avZ**2) ## eV
        self.E0       = 10. # eV
        self.E1       = self.mat.I
        # self.E1       = (self.mat.I/(self.E2**self.f2))**(1./self.f1)
        self.a0       = 42 ## [1/cm?]
        self.fw       = 4 ## [?]
        self.w1       = self.mat.Tc/self.E0
        self.ncontmax = 8 ## the maximum **mean** number of collisions in a step for
                          ## Poisson sampling of number of the actual number of collisions,
                          ## where the actual E-loss is this number times the associated energy.
                          ## Otherwise the actual E-loss is sampled directly from a Gaussian)
        self.Emax     = 1e308
        self.EkinMin  = -1
        self.EkinMax  = -1
        self.fmax     = -1
    
    # def __str__(self):
    #     return f"{self.name}"
    
    def getG4dEdx(self,E):
        # return self.tables.dEdxGraph.Eval(E*U.eV2MeV)*U.MeV2eV*1/(U.mm2cm) # eV/cm
        return self.tbl.dEdxGraph_eV_per_cm.Eval(E) # eV/cm

    ### definition in Equation 33.6 from PDG: https://pdg.lbl.gov/2016/reviews/rpp2016-rev-passage-particles-matter.pdf
    def delta(self,E):
        g = self.gamma(E)
        b = self.beta(E)
        # return 2*math.log(self.mat.Ep/self.mat.I) + 2*math.log(b*g) - 1
        cden  = self.mat.namedden["Cdensity"]
        mden  = self.mat.namedden["Mdensity"]
        aden  = self.mat.namedden["Adensity"]
        x0den = self.mat.namedden["X0density"]
        x1den = self.mat.namedden["X1density"]
        d = 0
        x = math.log((b*g)**2)/C.twoln10
        if(x<x0den): d = 0.0
        else: 
            d = C.twoln10*x - cden
            if(x<x1den): d += aden*math.pow((x1den-x),mden)
        return d

    ### the general Bethe-Bloch in eq 34.5 from https://pdg.lbl.gov/2023/reviews/rpp2023-rev-passage-particles-matter.pdf
    def BB(self,E,T):
        g = self.gamma(E)
        b = self.beta(E)
        XX = C.K*(self.z**2)*self.mat.ZoA*(1/b**2)
        YY = 0.5*math.log(2*C.me*((b*g)**2)*T/self.mat.I**2) - b**2 - self.delta(E)/2
        # YY = 0.5*math.log(2*me*((b*g)**2)*Tup/self.mat.I**2) - b**2
        return XX*YY*self.mat.rho ## eV/cm
        # return XX*YY*self.mat.rho if(T!=self.Wmax(E)) else self.fsec*XX*YY*self.mat.rho ## eV/cm
    
    ### xi function from PDG: https://pdg.lbl.gov/2016/reviews/rpp2016-rev-passage-particles-matter.pdf
    def xi_PDG(self,E,x): ## x should be in cm
        b = self.beta(E)
        X = x*self.mat.rho ## units of X: g/cm2
        return (C.K/2)*self.mat.ZoA*(self.z**2)*(X/(b**2)) # K is already converted to eV
    
    def gamma(self,E):
        tau  = E/self.m
        # taul = (2.*U.MeV2eV)/C.mp; ## lower limit of Bethe-Bloch formula: 2MeV/proton_mass
        # if(tau<taul): tau = taul ## It is not normal case for this function for low energy parametrisation have to be applied
        return tau+1. #(E + self.m)/self.m ## both in eV
    
    def beta(self,E):
        g = self.gamma(E)
        return math.sqrt(1-1/g**2)
    
    ### total energy
    def Etot(self,E):
        return E+self.m
    
    ### definition in Equation 33.4 from PDG: https://pdg.lbl.gov/2016/reviews/rpp2016-rev-passage-particles-matter.pdf
    def Wmax(self,E):
        g = self.gamma(E)
        b = self.beta(E)
        return ((2*C.me)*((b*g)**2))/(1 + 2*g*C.me/self.m + (C.me/self.m)**2)
    
    ### the production threshold for delta ray or the maximum energy transfer if this value smaller than the production threshold
    def Tup(self,E):
        Tmax = self.Wmax(E) ## maximum energy transfer 
        Tcut = self.mat.Tc  ## production threshold for delta ray 
        return Tmax if(Tmax<Tcut) else Tcut
    
    ### thik absorbers conditions
    def isThick(self,E,x):
        dEmean = x*self.dEdx(E,False)
        Tmax   = self.Wmax(E)
        return (self.m>C.me and dEmean>=self.NminBohr*self.mat.Tc and Tmax<=2.*self.mat.Tc)

    def isGauss(self,E,x,i):
        ni = -1
        if(i==1): ni = self.n12_mean(E,x,1)
        if(i==2): ni = self.n12_mean(E,x,2)
        if(i==3): ni = self.n3_mean(E,x)
        if(i==0): ni = self.n_0dE_mean(E,x)
        return (ni>self.ncontmax)

    def scaling(self,doScale=True):
        return min((1.0+500/self.mat.Tc),1.5) if(doScale) else 1.0

    ### the stopping power in eV/cm given
    ### (1) the production threshold (Tcut) for delta ray, or
    ### (2) the value directly from GEANT4 which usually corresponds to Tcut from (1) but it includes also the Bragg part
    def dEdx(self,E,doScale=True):
        scl = self.scaling(doScale)
        if(self.dedxmod=="BB:Tcut"): return self.BB(E,self.mat.Tc)/scl
        if(self.dedxmod=="G4:Tcut"): return self.getG4dEdx(E)/scl
        return -999

    ### modify the mean loss in case of "long" steps
    def modMeanLoss(self,E,x):
        #TODO: do I need to scale it as in the dEdx function above?
        meanLoss = x*self.dEdx(E,False)      ### eV original meanLoss without scaling, as is from the dEdx table
        modLoss  = self.tbl.getMeanLoss(E,x) ### eV modified meanLoss without scaling, corrected for range etc.
        return meanLoss,modLoss

    def meanLossFactor(self,E,x):
        #TODO: do I need to scale it as in the dEdx function above?
        meanLoss,modfLoss = self.modMeanLoss(E,x)
        return modfLoss/meanLoss if(meanLoss>0) else 1.

    ##########################################################################
    ### default Energy Loss Fluctuations model used in main Physics List: https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsReferenceManual/html/electromagnetic/energy_loss/fluctuations.html#id230
    def Sigma12(self,E,i):
        g  = self.gamma(E)
        b  = self.beta(E)
        f  = self.f1 if(i==1) else self.f2
        dE = self.E1 if(i==1) else self.E2 # eV
        XX = (f/dE) if(dE>0) else 0 # 1/eV
        YY = math.log((2*C.me)*((b*g)**2)/dE)-b**2
        ZZ = math.log((2*C.me)*((b*g)**2)/self.mat.I)-b**2
        return self.dEdx(E)*XX*(YY/ZZ)*(1.-self.r) # 1/cm
    def Sigma3(self,E):
        return self.dEdx(E)*((self.mat.Tc-self.E0)/(self.E0*self.mat.Tc*math.log(self.mat.Tc/self.E0)))*self.r # 1/cm
    
    ### mean number of interactions/collisions
    def n12_mean(self,E,x,i):
        return x*self.Sigma12(E,i)*self.meanLossFactor(E,x) ## dimensionless
    def n3_mean(self,E,x):
        return x*self.Sigma3(E)*self.meanLossFactor(E,x) ## dimensionless
    
    ### there are steps with only 1 or 0 secondaries
    ### TODO
    def isSecondary(self,E):
        self.EkinMin = min(self.mat.Tc,self.Wmax(E))
        self.EkinMax = min(self.Wmax(E),self.Emax)
        self.fmax = 1+0.5*(self.EkinMax/self.Etot(E))*(self.EkinMax/self.Etot(E)) if(self.spin>0) else 1
        if(self.EkinMin>=self.EkinMax): return False
        return True
    
    ### default Energy Loss Fluctuations model used in main Physics List: https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsReferenceManual/html/electromagnetic/energy_loss/fluctuations.html#id230
    def g_of_dE_integral1Tcut(self,E):
        return (self.E0*self.mat.Tc/(self.mat.Tc-self.E0)) * math.log(self.mat.Tc/self.E0) # eV
    def g_of_dE_integral2Tcut(self,E):
        return self.E0*self.mat.Tc # eV^2
    
    def RescaleS1(self,E,x):
        a1 = 0
        if(self.mat.Tc<=self.E1):
            return a1,self.E1 ### TODO: Was missing before 9/7/24, but this condition should usually be false anyway...
        S1 = self.Sigma12(E,1)
        ### modify meanLoss
        S1 *= self.meanLossFactor(E,x)
        ### the mean number of collisions <n1>
        a1 = S1*x
        ### rescale
        S1new = 0
        E1new = 0
        if(a1<self.a0):
            fwnow = 0.1+(self.fw-0.1)*math.sqrt(a1/self.a0)
            S1new = S1/fwnow
            E1new = self.E1*fwnow
        else:
            S1new = S1/self.fw
            E1new = self.E1*self.fw
        return S1new,E1new
    
    def Moment1(self,E,x,proc="EX1:EX2:ION"): # this is the mean
        # S1 = self.Sigma12(E,1)*self.E1 ## eV/cm
        s1,e1 = self.RescaleS1(E,x)
        S1 = s1*e1 ## eV/cm
        S2 = self.Sigma12(E,2)*self.E2 ## eV/cm
        S3 = self.Sigma3(E)*self.g_of_dE_integral1Tcut(E) ## eV/cm
        ### modify meanLoss
        sf = self.meanLossFactor(E,x)
        S2 *= sf ## S1 is already scaled
        S3 *= sf ## S1 is already scaled
        ### the moment
        M1 = 0
        if("EX1" in proc): M1 += S1 ## eV/cm
        if("EX2" in proc): M1 += S2 ## eV/cm
        if("ION" in proc): M1 += S3 ## eV/cm
        return M1
        
    def Moment2(self,E,x,proc="EX1:EX2:ION"): # this is the variance
        # S1 = self.Sigma12(E,1)*(self.E1**2) ## eV^2/cm
        s1,e1 = self.RescaleS1(E,x)
        S1 = s1*(e1**2) ## eV^2/cm
        S2 = self.Sigma12(E,2)*(self.E2**2) ## eV^2/cm
        S3 = self.Sigma3(E)*self.g_of_dE_integral2Tcut(E) ## eV^2/cm
        ### modify meanLoss
        sf = self.meanLossFactor(E,x)
        S2 *= sf ## S1 is already scaled
        S3 *= sf ## S1 is already scaled
        ### the moment
        M2 = 0
        if("EX1" in proc): M2 += S1 ## eV^2/cm
        if("EX2" in proc): M2 += S2 ## eV^2/cm
        if("ION" in proc): M2 += S3 ## eV^2/cm
        return M2

    def Mean(self,E,x,proc="EX1:EX2:ION"):
        return x*self.Moment1(E,x,proc) ## cm * eV/cm = eV
    
    ### default Energy Loss Fluctuations model used in main Physics List: https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsReferenceManual/html/electromagnetic/energy_loss/fluctuations.html#id230    
    def Width(self,E,x,proc="EX1:EX2:ION"): ###
        return math.sqrt(x*self.Moment2(E,x,proc)) ## sqrt(cm * eV^2/cm) = eV

    ### for thick media the model is Gaussian
    def MeanThick(self,E,x):
        meanLoss,modLoss = self.modMeanLoss(E,x)
        return modLoss

    ### https://geant4.kek.jp/lxr/source/processes/electromagnetic/standard/src/G4UniversalFluctuation.cc#L129
    ### Gaussian sigma for thick media
    def WidthThick(self,E,x):
        b  = self.beta(E)
        Tmax = self.Wmax(E)
        return math.sqrt( (Tmax/(b**2)-0.5*self.mat.Tc) * (C.twopi*C.me*C.re2) * x * (self.z**2) * self.mat.electronDensity )

    #########################################
    ### get all necessary pars to build the model shape
    def GetModelPars(self,E,x):
        point = ("%.2f" % (E*U.eV2MeV))+"MeV_"+("%.7f" % (x*U.cm2um))+"um"
        scl   = self.scaling()
        meanLoss,modLoss = self.modMeanLoss(E,x)
        pars = {"point":point, "build":"NONE", "scale":scl, "param":{}}
        pars["param"].update({"dx":x})
        pars["param"].update({"E":E})
        pars["param"].update({"primprt":self.primprt})
        pars["param"].update({"minLoss":self.minloss})
        pars["param"].update({"meanLoss":meanLoss}) ## the original meanLoss without scaling
        pars["param"].update({"modLoss":modLoss}) ## the modified meanLoss without scaling
        pars["param"].update({"Tcut":self.mat.Tc}) ## for secondaries
        pars["param"].update({"Tmax":self.Wmax(E)}) ## for secondaries
        pars["param"].update({"Etot":self.Etot(E)}) ## for secondaries
        pars["param"].update({"b2":self.beta(E)**2}) ## for secondaries
        pars["param"].update({"w3":-1}) ## Borysov ion
        pars["param"].update({"w":-1}) ## Borysov ion
        pars["param"].update({"p3":-1}) ## Borysov ion
        pars["param"].update({"e1":-1}) ## Borysov ex1
        pars["param"].update({"n1":-1}) ## Borysov ex1
        pars["param"].update({"e2":-1}) ## Borysov ex2
        pars["param"].update({"n2":-1}) ## Borysov ex2
        pars["param"].update({"ion_mean":-1}) ## Gauss ion
        pars["param"].update({"ion_sigma":-1}) ## Gauss ion
        pars["param"].update({"ex1_mean":-1}) ## Gauss ex1
        pars["param"].update({"ex1_sigma":-1}) ## Gauss ex1
        pars["param"].update({"ex2_mean":-1}) ## Gauss ex2
        pars["param"].update({"ex2_sigma":-1}) ## Gauss ex2
        pars["param"].update({"thk_mean":-1})  ## Thick limit for Gauss and Gamma
        pars["param"].update({"thk_sigma":-1}) ## Thick limit for Gauss
        pars["param"].update({"thk_neff":-1})  ## Thick limit for Gamma  
        pars["param"].update({"EkinMin":-1}) ## secondaries
        pars["param"].update({"EkinMax":-1}) ## secondaries
        pars["param"].update({"fmax":-1}) ## secondaries
        
        SECB = False
        BEBL = False
        TGAU = False
        TGAM = False
        EX1G = False
        EX2G = False
        IONG = False
        EX1B = False
        EX2B = False
        IONB = False
        
        ######################
        ### Tiny loss models.
        ### note that (1) meanLoss is without scaling, and (2) Tcut should not be smaller than E0(=10 eV)
        # if(pars["param"]["meanLoss"]<self.minloss or self.mat.Tc<=self.E0):
        if(pars["param"]["modLoss"]<self.minloss or self.mat.Tc<=self.E0):
            pars["build"] = "BEBL"
            BEBL = True
            ######################
            ### Secondaries
            if(self.isSecondary(E)):
                pars["build"] += "->SEC.B"
                pars["param"]["EkinMin"] = self.EkinMin
                pars["param"]["EkinMax"] = self.EkinMax
                pars["param"]["fmax"]    = self.fmax
                SECB =  True
            ######################
            ### return
            return pars


        ######################
        ### Thick models
        if(self.isThick(E,x)):
            mua  = self.MeanThick(E,x)
            siga = self.WidthThick(E,x)
            sn   =  mua/siga
            neff = sn*sn
            pars["param"]["thk_mean"] = mua
            if(sn>2.):
                pars["param"]["thk_sigma"] = siga
                pars["build"] = "THK.GAUSS"
                TGAU = True
            else:
                pars["param"]["thk_neff"] = neff
                pars["build"] = "THK.GAMMA"
                TGAM = True
            ######################
            ### Secondaries
            if(self.isSecondary(E)):
                pars["build"] += "->SEC.B"
                pars["param"]["EkinMin"] = self.EkinMin
                pars["param"]["EkinMax"] = self.EkinMax
                pars["param"]["fmax"]    = self.fmax
                SECB =  True
            ######################
            ### return
            return pars
            

        ######################
        ### Thin models
        s1,e1 = self.RescaleS1(E,x)
        n1    = s1*x
        n3    = self.n3_mean(E,x)
        if(n1<=0.): n3 /= self.r ### TODO: Was missing before 9/7/24, but this condition should normally be false...
        
        
        ### excitation of type 1
        if(self.f1>0 and n1>0):
            if(self.isGauss(E,x,1)):
                pars["param"]["ex1_mean"]  = self.Mean(E,x,proc="EX1")
                pars["param"]["ex1_sigma"] = self.Width(E,x,proc="EX1")
                EX1G = True if(pars["param"]["ex1_sigma"]>0) else False ### TODO: Was missing before 9/7/24, but this condition should normally be false...
            else:
                # s1,e1 = self.RescaleS1(E,x)
                # n1 = s1*x
                pars["param"]["e1"] = e1
                pars["param"]["n1"] = n1
                EX1B = True if(n1>0) else False
        #########################
        # ### excitation of type 2
        # if(self.f2>0):
        #     if(self.isGauss(E,x,2)):
        #         pars["param"]["ex2_mean"]  = self.Mean(E,x,proc="EX2")
        #         pars["param"]["ex2_sigma"] = self.Width(E,x,proc="EX2")
        #         EX2G = True
        #     else:
        #         s2,e2 = self.RescaleS1(E,x)
        #         n2 = s2*x
        #         pars["param"]["e2"] = e2
        #         pars["param"]["n2"] = n2
        #         EX2B = True
        ##########################
        ### Ionization
        if(n3>0):
            alpha  = 1.
            naAvg  = 0.
            alpha1 = 0.
            p3     = n3
            w3     = alpha*self.E0
            ### gaussian part (conditional)
            if(self.isGauss(E,x,3)):
                alpha  = (self.w1*(self.ncontmax+n3))/(self.w1*self.ncontmax+n3)
                alpha1 = alpha*math.log(alpha)/(alpha-1.)
                naAvg  = n3*self.w1*(alpha-1)/(alpha*(self.w1-1.))
                p3     = n3 - naAvg
                w3     = alpha*self.E0
                pars["param"]["ion_mean"]  = naAvg*alpha1*self.E0 ### TODO: as in G4
                pars["param"]["ion_sigma"] = math.sqrt(naAvg*(alpha-alpha1**2)*(self.E0**2))
                IONG = True if(pars["param"]["ion_sigma"]>0) else False ### TODO: Was missing before 9/7/24, but this condition should normally be false...
            ### poisson part (~always)
            if(self.mat.Tc>w3):
                w = (self.mat.Tc-w3)/self.mat.Tc
                pars["param"]["w3"] = w3
                pars["param"]["p3"] = p3
                pars["param"]["w"]  = w
                IONB = True

        #######################
        ### finally, build string
        if(IONB and IONG and EX1G):                  pars["build"] = "ION.B->ION.G->EX1.G"
        if(IONB and EX1B and IONG):                  pars["build"] = "ION.B->EX1.B->ION.G"
        if(IONB and EX1B and not IONG and not EX1G): pars["build"] = "ION.B->EX1.B"
        ######################
        ### Secondaries
        if(self.isSecondary(E)):
            pars["build"] += "->SEC.B"
            pars["param"]["EkinMin"] = self.EkinMin
            pars["param"]["EkinMax"] = self.EkinMax
            pars["param"]["fmax"]    = self.fmax
            SECB =  True

        ######################
        ### return
        return pars

