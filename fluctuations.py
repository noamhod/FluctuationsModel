import array
import math
import numpy as np
# import scipy.integrate as integrate
# import scipy as sp
import ROOT
from ROOT import TH1D, TH2D, TMath, TF1, TCanvas, TLegend, TGraph
import units as U
import constants as C
import material as mat
import bins
import shapes

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)


#####################################################################
#####################################################################
#####################################################################


class Parameters:
    def __init__(self,name,mass,charge,material,dedxmodel,bbtable,bbfunc):
        self.name    = name
        self.m       = mass
        self.z       = charge
        self.mat     = material
        self.dedxmod = dedxmodel
        if(self.dedxmod!="BB:Tcut" and self.dedxmod!="BB:Tmax" and self.dedxmod!="G4:Tcut"):
            print(f"Unknown model named {self.dedxmod}. Quitting")
            quit()
        print(f"Using dE/dx model: {self.dedxmod}")
        
        ### default Energy Loss Fluctuations model used in main Physics List:
        self.r        = 0.56 #0.55 ## rate (fraction) of the ionization part of the total loss (excitation+ionization)
        self.kap      = 10 ## lower limit of the number of interactions of the particle in a step
        self.minloss  = 10. ## eV
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
        self.ncontmax = 8 ## the maximum **mean** number of collisions in a step for
                          ## Poisson sampling of number of the actual number of collisions,
                          ## where the actual E-loss is this number times the associated energy.
                          ## Otherwise the actual E-loss is sampled directly from a Gaussian)

        self.gBB = self.setG4BBdEdxFromTable(bbtable)
        self.hBBlow,self.hBBhig = self.setG4BBdEdx(bbfunc)
    
    def __str__(self):
        return f"{self.name}"
        
    def setG4BBdEdxFromTable(self,fname):
        hname = fname
        hname = hname.split(".")[0]
        arr_E    = array.array( 'd' )
        arr_dEdx = array.array( 'd' )
        with open(fname) as f:
            for line in f:
                if("#" in line): continue
                line = line.replace("\n","")
                words = line.split("   ")
                arr_E.append( float(words[0]) )
                arr_dEdx.append( float(words[1]) )
        npts = len(arr_E)
        print(f"Read {npts} points from file {fname}")
        gBB = TGraph(npts,arr_E,arr_dEdx)
        gBB.SetLineColor(ROOT.kBlue)
        gBB.GetXaxis().SetTitle("E [MeV]")
        gBB.GetYaxis().SetTitle("dE/dx [MeV/mm]")
        return gBB

    def setG4BBdEdx(self,fname):
        hname = fname
        hname = hname.split(".")[0]
        arr_E        = []
        arr_dEdx_low = []
        arr_dEdx_hig = []
        with open(fname) as f:
            for line in f:
                if("#" in line): continue
                words = line.split(",")
                arr_E.append( float(words[0]) )
                arr_dEdx_low.append( float(words[1]) )
                arr_dEdx_hig.append( float(words[2]) )
        npts = len(arr_E)
        emin = arr_E[0]
        emax = arr_E[npts-1]
        de   = arr_E[1]-arr_E[0]
        print(f"Read {npts} points from file {fname}")
        hBBlow = TH1D(hname+"_low",";E [MeV];dE/dx [MeV*cm^{2}/g]",npts,emin-de/2,emax+de/2)
        hBBhig = TH1D(hname+"_hig",";E [MeV];dE/dx [MeV*cm^{2}/g]",npts,emin-de/2,emax+de/2)
        hBBlow.SetLineColor(ROOT.kRed)
        hBBhig.SetLineColor(ROOT.kBlack)
        for i,E in enumerate(arr_E):
            b = hBBlow.FindBin(E)
            hBBlow.SetBinContent(b, arr_dEdx_low[i])
            hBBhig.SetBinContent(b, arr_dEdx_hig[i])
        return hBBlow,hBBhig
    
    def getG4BBdEdx(self,E):
        return self.gBB.Eval(E*U.eV2MeV)*U.MeV2eV*1/(U.mm2cm) # eV/cm
    
    def correctG4BBdEdx(self,E,x):
        meanLoss = x*self.getG4BBdEdx(E)
        if(self.mat.Tc<=self.E0): return meanLoss ## very small step or low-density material
        scaling = min(1.+(0.5*U.keV2eV)/self.mat.Tc, 1.50)
        meanLoss /= scaling
        return meanLoss if(meanLoss>self.minloss) else self.minloss

    ### https://geant4.kek.jp/lxr/source/processes/electromagnetic/standard/src/G4UniversalFluctuation.cc#L129
    # def sigmaThick(self,E,x):
    #     b    = self.beta(E)
    #     Tmax = self.Wmax(E)
    #     return math.sqrt( (Tmax/(b**2) - 0.5*self.mat.Tc) * (C.twopi*(re**2)) * x * (self.z**2) * self.mat.electronDensity )

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
    
    ### MPV parameter
    ### definition in Equation 33.11 from PDG: https://pdg.lbl.gov/2016/reviews/rpp2016-rev-passage-particles-matter.pdf
    def Delta_p_PDG(self,E,x): ## units of x are cm
        g = self.gamma(E)
        b = self.beta(E)
        XI = self.xi_PDG(E,x) # eV
        AA = math.log( (2 * C.me * (b*g)**2) / self.mat.I )
        BB = math.log(XI/self.mat.I)
        CC = C.j - b**2 - self.delta(E)
        return XI*(AA+BB+CC) # eV

    def gamma(self,E):
        tau = E/self.m
        taul = (2.*U.MeV2eV)/C.mp; ## lower limit of Bethe-Bloch formula: 2MeV/proton_mass
        if(tau<taul): tau = taul ## It is not normal case for this function for low energy parametrisation have to be applied
        return tau+1. #(E + self.m)/self.m ## both in eV
    
    def beta(self,E):
        g = self.gamma(E)
        return math.sqrt(1-1/g**2)
    
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
        dEmean = self.correctG4BBdEdx(E,x)
        Tmax   = self.Wmax(E)
        return (self.m>C.me and dEmean>self.kap*self.mat.Tc and Tmax<=2.*self.mat.Tc)

    def isGauss(self,E,x,i):
        ni = -1
        if(i==1):
            ni = self.n12_mean(E,x,1)
            # s1,e1 = self.RescaleS1(E,x)
            # ni = s1*x
        if(i==2): ni = self.n12_mean(E,x,2)
        if(i==3): ni = self.n3_mean(E,x)
        if(i==0): ni = self.n_0dE_mean(E,x)
        return (ni>self.ncontmax)

    ### the stopping power in eV/cm given
    ### (1) the production threshold (Tcut) for delta ray, or
    ### (2) the maximum energy transfer (Tmax), or
    ### (3) the value directly from GEANT4 which usually corresponds to Tcut from (1)
    def dEdx(self,E,doScale=True):
        # scl = 1
        scl = min((1+500/self.mat.Tc),1.5) if(doScale) else 1
        if(self.dedxmod=="BB:Tcut"): return self.BB(E,self.mat.Tc)/scl
        if(self.dedxmod=="BB:Tmax"): return self.BB(E,self.Wmax(E))/scl
        if(self.dedxmod=="BB:Tup"):  return self.BB(E,self.Tup(E))/scl
        if(self.dedxmod=="G4:Tcut"): return self.getG4BBdEdx(E)/scl
        return -999

    def getmpv(self,mu):
        # return math.floor(mu) if(mu>1) else 0 ## dimensionless
        return math.floor(mu) if(mu>1) else mu ## dimensionless

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
    def Sigma4(self,E):
        return self.dEdx(E)*(1./self.mat.Tc) # 1/cm #TODO: this is just dimensionally-guessed... 
    def Sigma0(self,E):
        return self.dEdx(E)*(1./self.E0) # 1/cm
    
    ### mean number of interactions/collisions
    def n12_mean(self,E,x,i):
        return x*self.Sigma12(E,i) ## dimensionless
    def n3_mean(self,E,x):
        return x*self.Sigma3(E) ## dimensionless
    def n_0dE_mean(self,E,x):
        return x*self.Sigma0(E) ## dimensionless
    def n12_mpv(self,E,x,i):
        mu = self.n12_mean(E,x,i)
        return self.getmpv(mu) ## dimensionless
    def n3_mpv(self,E,x):
        mu = self.n3_mean(E,x)
        return self.getmpv(mu) ## dimensionless
    def n_0dE_mpv(self,E,x):
        mu = self.n_0dE_mean(E,x)
        return self.getmpv(mu) ## dimensionless
        
    def is0dE(self,E,x):   
        n1 = self.n12_mean(E,x,1)
        n2 = self.n12_mean(E,x,2)
        n3 = self.n3_mean(E,x)
        P_0dE = math.exp(-(n1+n2+n3))
        return (P_0dE>0.01)
    
    ### there are steps with only 1 or 0 secondaries
    def isSecondary(self,E,x):
        return (self.dEdx(E)*x>self.mat.Tc)
        # return ( (self.BB(E,self.Wmax(E))-self.BB(E,self.mat.Tc))*x>self.mat.Tc )
        
    def elossSecondary(self,E,x):
        return self.mat.Tc if(self.isSecondary(E,x)) else 0.
        # return (self.Wmax(E)-self.mat.Tc)/2 if(self.isSecondary(E,x)) else 0.
    
    ### default Energy Loss Fluctuations model used in main Physics List: https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsReferenceManual/html/electromagnetic/energy_loss/fluctuations.html#id230
    def g_of_dE_integral1Tup(self,E):
        return (self.E0*self.Tup(E)/(self.Tup(E)-self.E0)) * math.log(self.Tup(E)/self.E0) # eV
    def g_of_dE_integral1Tcut(self,E):
        return (self.E0*self.mat.Tc/(self.mat.Tc-self.E0)) * math.log(self.mat.Tc/self.E0) # eV
    def g_of_dE_integral1Tmax(self,E):
        return (self.mat.Tc*self.Wmax(E)/(self.Wmax(E)-self.mat.Tc)) * math.log(self.Wmax(E)/self.mat.Tc) # eV
    def g_of_dE_integral2Tup(self,E):
        return self.E0*self.Tup(E) # eV^2
    def g_of_dE_integral2Tcut(self,E):
        return self.E0*self.mat.Tc # eV^2
    def g_of_dE_integral2Tmax(self,E):
        return self.mat.Tc*self.Wmax(E) # eV^2
    def g_of_dE_integral2TupAlpha(self,E,alpha):
        return (self.E0*alpha)*self.Tup(E) # eV^2
    
    # def RescaleS1(self,E,x):
    #     S1 = self.Sigma12(E,1)
    #     a1 = S1*x
    #     if(a1<self.a0):
    #         fwnow = 0.1+(self.fw-0.1)*math.sqrt(a1/self.a0)
    #         S1 = S1/fwnow
    #         E1 = self.E1*fwnow
    #     else:
    #         S1 = S1/self.fw
    #         E1 = self.E1*self.fw
    #     return S1,E1
    
    def Moment1(self,E,x,proc="EX1:EX2:ION:SEC:ZER"): # this is the mean
        S1 = self.Sigma12(E,1)*self.E1 ## eV/cm
        # s1,e1 = self.RescaleS1(E,x)
        # S1 = s1*e1 ## eV/cm
        S2 = self.Sigma12(E,2)*self.E2 ## eV/cm
        S3 = self.Sigma3(E)*self.g_of_dE_integral1Tup(E) ## eV/cm
        S4 = self.elossSecondary(E,x) ## eV/cm #TODO!!!! UNITS
        S0 = self.Sigma0(E)*self.E0 ## self.Sigma0(E)*self.g_of_dE_integral1(E) ## eV/cm
        M1 = 0
        if("EX1" in proc): M1 += S1 ## eV/cm
        if("EX2" in proc): M1 += S2 ## eV/cm
        if("ION" in proc): M1 += S3 ## eV/cm
        if("SEC" in proc): M1 += S4 ## eV/cm
        if("ZER" in proc and self.is0dE(E,x)): M1 = S0 ## eV/cm
        return M1
        
    def Moment2(self,E,x,proc="EX1:EX2:ION:SEC:ZER"): # this is the variance
        S1 = self.Sigma12(E,1)*(self.E1**2) ## eV^2/cm
        # s1,e1 = self.RescaleS1(E,x)
        # S1 = s1*(e1**2) ## eV^2/cm
        S2 = self.Sigma12(E,2)*(self.E2**2) ## eV^2/cm
        # S3 = self.Sigma3(E)*self.g_of_dE_integral2Tcut(E) ## eV^2/cm
        S3 = self.Sigma3(E)*self.g_of_dE_integral2Tcut(E) ## eV^2/cm
        S4 = (self.elossSecondary(E,x)**2) ## eV^2/cm #TODO: UITS
        S0 = self.Sigma0(E)*(self.E0**2) ## self.Sigma0(E)*self.g_of_dE_integral2(E) ## eV^2/cm
        M2 = 0
        if("EX1" in proc): M2 += S1 ## eV^2/cm
        if("EX2" in proc): M2 += S2 ## eV^2/cm
        if("ION" in proc): M2 += S3 ## eV^2/cm
        if("SEC" in proc): M2 += S4 ## eV^2/cm
        if("ZER" in proc and self.is0dE(E,x)): M2 = S0 ## eV^2/cm
        return M2
        
    def MPV(self,E,x,proc="EX1:EX2:ION:SEC:ZER"):
        mpv1 = self.n12_mpv(E,x,1)*self.E1 ## eV
        # s1,e1 = self.RescaleS1(E,x)
        # mpv1 = (self.getmpv(s1*x))*e1 ## eV
        mpv2 = self.n12_mpv(E,x,2)*self.E2 ## eV
        mpv3 = self.n3_mpv(E,x)*self.g_of_dE_integral1Tup(E) ## eV
        mpv4 = self.elossSecondary(E,x) ## eV
        mpv0 = self.n_0dE_mpv(E,x)*self.E0 ## self.n_0dE_mpv(E,x)*self.g_of_dE_integral1(E) ## eV
        mpv = 0
        if("EX1" in proc): mpv += mpv1 ## eV
        if("EX2" in proc): mpv += mpv2 ## eV
        if("ION" in proc): mpv += mpv3 ## eV
        if("SEC" in proc): mpv += mpv4 ## eV
        if("ZER" in proc and self.n_0dE_mpv(E,x)): mpv = mpv0 ## eV
        return mpv

    def Mean(self,E,x,proc="EX1:EX2:ION:SEC:ZER"):
        return x*self.Moment1(E,x,proc) ## cm * eV/cm = eV
    
    ### default Energy Loss Fluctuations model used in main Physics List: https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsReferenceManual/html/electromagnetic/energy_loss/fluctuations.html#id230    
    def Width(self,E,x,proc="EX1:EX2:ION:SEC:ZER"): ###
        return math.sqrt(x * self.Moment2(E,x,proc)) ## sqrt(cm * eV^2/cm) = eV

    #########################################
    ### for thick media the model is Gaussian
    def MeanThick(self,E,x):
        return self.correctG4BBdEdx(E,x)

    ### https://geant4-userdoc.web.cern.ch/UsersGuides/PhysicsReferenceManual/html/electromagnetic/energy_loss/fluctuations.html#id230
    ### Gaussian sigma for thick media
    def WidthThick(self,E,x):
        b  = self.beta(E)
        Tmax = self.Wmax(E)
        Omega2 = C.twopi*(C.re**2)*C.me*self.mat.electronDensity*((self.z/b)**2)*Tmax*x*(1-((b**2)/2.)*self.mat.Tc/Tmax)
        ### https://indico.cern.ch/event/102427/contributions/11182/attachments/8373/12423/ionization.pdf
        # Omega2 = C.twopi*(C.re**2)*C.me*self.mat.electronDensity*((self.z/b)**2)*self.mat.Tc*x*(1-((b**2)/2.))
        return math.sqrt(Omega2)

    #########################################
    ### the model: thin (Landau) / thick (Gaus)
    def Model(self,E,x):
        loc = -1
        wdt = -1
        mod = ""
        if(self.isThick(E,x)):
            loc = self.MeanThick(E,x)
            wdt = self.WidthThick(E,x)
            mod = "Gaus"
        else:
            loc = self.MPV(E,x)
            wdt = self.Width(E,x)
            mod = "Landau"
        return loc,wdt,mod

    def DifferentialModel(self,E,x,doSec=True):
        meanG = 0
        variG = 0
        mpvL  = 0
        variL = 0
        Gauss  = False
        Landau = False
        ### excitation of type 1
        if(self.isGauss(E,x,1)):
            meanG += self.Mean(E,x,proc="EX1")
            variG += self.Width(E,x,proc="EX1")**2
            if(variG>0): Gauss = True
            print(f"Gauss EX1: mean={meanG}, variance={variG}")
        else:
            mpvL  += self.MPV(E,x,proc="EX1")
            # variL += self.Width(E,x,proc="EX1")**2
            variL += self.n12_mean(E,x,1)*(self.E1**2) ### poisson's variance equals to the mean
            if(variL>0): Landau = True
            print(f"Landau EX1: mpv={mpvL}, variance={variL}")
        ### excitation of type 2 --> should be no such contribution if E1=I
        if(self.isGauss(E,x,2)):
            meanG += self.Mean(E,x,proc="EX2")
            variG += self.Width(E,x,proc="EX2")**2
            if(variG>0): Gauss = True
            print(f"Gauss EX2: mean={meanG}, variance={variG}")
        else:
            mpvL  += self.MPV(E,x,proc="EX2")
            # variL += self.Width(E,x,proc="EX2")**2
            variL += self.n12_mean(E,x,2)*(self.E2**2) ### poisson's variance equals to the mean
            if(variL>0): Landau = True
            print(f"Landau EX2: mpv={mpvL}, variance={variL}")

        ### ionization
        # if(self.isGauss(E,x,3)):
        #     meanG += self.Mean(E,x,proc="ION")
        #     variG += self.Width(E,x,proc="ION")**2
        #     if(variG>0): Gauss = True
        # mpvL  += self.MPV(E,x,proc="ION")
        # variL += self.Width(E,x,proc="ION")**2
        # if(variL>0): Landau = True
        
        ### ionization 
        alpha  = 1.
        naAvg  = 0.
        alpha1 = 0.
        n3     = self.n3_mean(E,x)
        ### gaussian part (conditional)
        if(self.isGauss(E,x,3)):
            w1 = self.mat.Tc/self.E0
            alpha  = (w1*(self.ncontmax+n3))/(w1*self.ncontmax+n3)
            alpha1 = alpha*math.log(alpha)/(alpha-1)
            naAvg  = n3*w1*(alpha-1)/(alpha*(w1-1))
            meanG += (self.Mean(E,x,proc="ION") + naAvg*self.E0*alpha1)
            variG += self.Width(E,x,proc="ION")**2 + naAvg*(alpha-alpha1**2)*(self.E0**2)
            if(variG>0): Gauss = True
            print(f"Gauss ION: mean={meanG}, variance={variG}")
        ### poisson part (~always)
        w3 = alpha*self.E0
        if(self.mat.Tc>w3):
            p3 = n3 - naAvg
            # w = (self.mat.Tc-w3)/self.mat.Tc
            # mpvL  += self.getmpv(p3) #self.MPV(E,x,proc="ION")
            # variL += p3*self.g_of_dE_integral2Tup(E) ### poisson's variance equals to the mean
            mpvL  += self.MPV(E,x,proc="ION")
            variL += self.Width(E,x,proc="ION") ### poisson's variance equals to the mean
            if(variL>0): Landau = True
            print(f"Landau ION: mpv={mpvL}, variance={variL}")
        
        #############
        #TODO !!!
        ### secondaries should always be there?
        if(doSec):
            meanG += self.Mean(E,x,proc="SEC")
            variG += self.Width(E,x,proc="SEC")**2
            mpvL  += self.MPV(E,x,proc="SEC")
            # variL += self.Width(E,x,proc="SEC")**2
            variL += self.elossSecondary(E,x)**2 ### poisson's variance equals to the mean #TODO
        
        ### construct the model
        widthG = math.sqrt(variG)
        widthL = math.sqrt(variL)
        model = {}
        if(Gauss):  model.update({"Gauss":{"mean":meanG, "width":widthG}})
        if(Landau): model.update({"Landau":{"mpv":mpvL, "width":widthL}})
        return model
    
    
    def BBlowE(self,E,T):
        g = self.gamma(E)
        b = self.beta(E)
        Tup = self.Tup(E)
        tau = E/self.m ## tau is relative energy
        taul = 2.*U.MeV2eV/C.mp; ## lower limit of Bethe-Bloch formula: 2MeV/proton_mass
        rateMass = C.me/self.m
        bg2lim = 0.0169
        taulim = 8.4146e-3
        ## It is not normal case for this function for low energy parametrisation have to be applied
        if(tau<taul): tau = taul
        
        eexc  = self.mat.namedden["MeanIonisationPotential"]*U.eV2MeV ## material->GetIonisation()->GetMeanExcitationEnergy(); #TODO is this the same thing??
        eexc2 = eexc*eexc
        cden  = self.mat.namedden["Cdensity"]
        mden  = self.mat.namedden["Mdensity"]
        aden  = self.mat.namedden["Adensity"]
        x0den = self.mat.namedden["X0density"]
        x1den = self.mat.namedden["X1density"]
        
        shellCorrectionVector = [0,0,0]
        for j in range(3):
            # shellCorrectionVector[j] = nAtomsPerVolume * self.shellcorrvec[j] * 2.0 / fMaterial->GetTotNbOfElectPerVolume()
            shellCorrectionVector[j] = self.mat.numberOfAtomsPerVolume * self.mat.shellcorrvec[j] * 2.0 / self.mat.electronDensity
        # print("shellCorrectionVector=",shellCorrectionVector)
        
        bg2   = tau*(tau+2.0)
        beta2 = bg2/(g*g)
        # tmax  = 2.*electron_mass_c2*bg2/(1.+2.*g*rateMass+rateMass*rateMass)
        # print("tmax=",tmax)
        # ionloss = math.log(2.0*C.me*bg2*tmax/eexc2)-2.0*beta2 #math.log(2.0*electron_mass_c2*bg2*tmax/eexc2)-2.0*beta2
        ionloss = math.log(2.0*C.me*bg2*T/eexc2)-2.0*beta2 #math.log(2.0*electron_mass_c2*bg2*tmax/eexc2)-2.0*beta2
        # print("ionloss=",ionloss)
        
        ### density correction     
        delta = 0
        x = math.log(bg2)/C.twoln10
        if(x<x0den): delta = 0.0
        else: 
            delta = C.twoln10*x - cden
            if(x<x1den): delta += aden*math.pow((x1den-x),mden)
        # print("delta=",delta)
        
        ### shell correction 
        sh = 0.0
        x  = 1.0
        if(bg2>bg2lim):
            for j in range(3):
                x *= bg2
                sh += shellCorrectionVector[j]/x
        else:
            for j in range(3):
                x *= bg2lim
                sh += shellCorrectionVector[j]/x
            sh *= math.log(tau/taul)/math.log(taulim/taul)
        # print("sh=",sh)
        
        ### now compute the total ionization loss 
        ionloss -= delta + sh
        # ionloss *= C.twopi_mc2_rcl2*electronDensity/beta2
        ionloss *= 0.5*(C.K*(self.z**2)*self.mat.ZoA/beta2)*self.mat.rho ## eV/cm ## TODO is this the same thing as the line above??
        
        if(ionloss<0.0): ionloss = 0.0
        return ionloss

    def Rsec(self,E):
        return (self.BB(E,self.Wmax(E))-self.BB(E,self.mat.Tc))/self.BB(E,self.Wmax(E))

#####################################################
#####################################################
#####################################################



def main():
    ### Tungsten parameters: https://pdg.lbl.gov/2020/AtomicNuclearProperties/HTML/tungsten_W.html
    rho_W = 19.30     # Tungsten, g/cm3
    Z_W   = [74]      # Tungsten atomic number (Z)
    A_W   = [183.841] # Tungsten atomic mass (A)
    I_W   = 727.0     # Tungsten mean excitation energy (I), eV
    Ep_W  = 80.32     # Tungsten plasma energy (E_p), eV
    Tc_W  = 990       # Tungsten production threshold for delta ray, eV
    den_W = [80.315, 1.997, 5.4059, 0.2167, 3.496, 0.15509, 2.8447, 0.14, 0.027, 727.]
    nel_W = 1
    W = mat.Material("Tungsten","W",rho_W,Z_W,A_W,I_W,Tc_W,den_W,nel_W)

    ### Aluminium
    rho_Al = 2.699     # Aluminum, g/cm3
    Z_Al   = [13]      # Aluminum atomic number (Z)
    A_Al   = [26.98]   # Aluminum atomic mass (A)
    I_Al   = 166.0     # Aluminum mean excitation energy (I), eV
    Ep_Al  = 32.86     # Aluminum plasma energy (E_p), eV
    Tc_Al  = 990       # Aluminum production threshold for delta ray, eV
    den_Al = [32.86, 2.18, 4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12, 0.061, 166.]
    nel_Al = 1
    Al = mat.Material("Aluminum","Al",rho_Al,Z_Al,A_Al,I_Al,Tc_Al,den_Al,nel_Al)

    ### Silicon parameters: https://pdg.lbl.gov/2022/AtomicNuclearProperties/HTML/silicon_Si.html
    rho_Si = 2.329     # Silicon, g/cm3
    Z_Si   = [14]      # Silicon atomic number (Z)
    A_Si   = [28.0855] # Silicon atomic mass (A)
    I_Si   = 173.0     # Silicon mean excitation energy (I), eV
    Ep_Si  = 31.05     # Silicon plasma energy (E_p), eV
    Tc_Si  = 990       # Silicon, production threshold for delta ray, eV
    den_Si = [31.055, 2.103, 4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14, 0.059, 173.]
    nel_Si = 1
    Si = mat.Material("Silicon","Si",rho_Si,Z_Si,A_Si,I_Si,Tc_Si,den_Si,nel_Si)

    dEdxModel = "BB:Tcut"
    par = Parameters("Silicon parameters",C.mp,+1,Si,dEdxModel,"eloss_p_si.txt","BB.csv")
    
    print(f"dEdx(2 MeV)={par.getG4BBdEdx(2*U.MeV2eV)*U.eV2MeV} [MeV/cm]")
    
    print(f"5um*dEdx = {(5*U.um2cm)*par.getG4BBdEdx(2*U.MeV2eV)*U.eV2MeV} [MeV]")
    print(f"corr = {par.correctG4BBdEdx(2*U.MeV2eV,5*U.um2cm)*U.eV2MeV} [MeV]")
    
    
    print(f"Wmax(100 MeV) = {par.Wmax(100*U.MeV2eV)*U.eV2MeV}")
    print(f"isThick(100 MeV, 5 um) = {par.isThick(100*U.MeV2eV,5*U.um2cm)}")
    
    print(f"isThick(5 MeV, 5 um) = {par.isThick(5*U.MeV2eV,5*U.um2cm)}")

    # hdE_fixed =  TH1D("hdE_fixed","E=100 MeV, dx=5 #mum;#DeltaE [MeV];Particles", len(bins.dEbins_small)-1,array.array("d",bins.dEbins_small))
    hdE_fixed =  TH1D("hdE_fixed","E=100 MeV, dx=5 #mum;#DeltaE [MeV];Particles", 10000,1e-4,0.5)
    with open("dEs.csv") as f:
        for line in f:
            if("#" in line): continue
            words = line.split(",")
            dE = float(words[1])*U.eV2MeV
            hdE_fixed.Fill(dE)
    # hdE_fixed.Scale(1./hdE_fixed.Integral(), "width")
    cnv = TCanvas("cnv","",500,500)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetGridx()
    hdE_fixed.SetLineColor(ROOT.kBlack)
    hdE_fixed.SetFillColorAlpha(ROOT.kBlack,0.35)
    hdE_fixed.Draw("hist")
    ### get the Landau for the MPV and Width
    func = shapes.Functions("Landau")
    Delta_p,Width,Model = par.Model(100*U.MeV2eV,5*U.um2cm)
    Delta_p = Delta_p*U.eV2MeV
    Width   = Width*U.eV2MeV
    Gwidth  = par.WidthThick(100*U.MeV2eV,5*U.um2cm)*U.eV2MeV
    ### the model
    function = func.fLandau(hdE_fixed.GetXaxis().GetXmin(),hdE_fixed.GetXaxis().GetXmax(),[Delta_p, Width, 1],"fixed")
    # function = func.fLandauGaus(hdE_fixed.GetXaxis().GetXmin(),hdE_fixed.GetXaxis().GetXmax(),[Width, Delta_p, 1, Gwidth],"fixed")
    hLandau = func.f2h(function,hdE_fixed)
    hdEmaximum = hdE_fixed.GetBinContent( hdE_fixed.GetMaximumBin() )
    hLandaumaximum = hLandau.GetBinContent( hLandau.GetMaximumBin() )
    hLandau.Scale(hdEmaximum/hLandaumaximum)
    hLandau.SetFillColorAlpha(ROOT.kRed,0.1)
    hLandau.Draw("hist same")
    
    # hLandauGuas = func.asH(hdE_fixed,hdE_fixed.GetXaxis().GetXmin(),hdE_fixed.GetXaxis().GetXmax(),[Width, Delta_p, 1, Gwidth],"LandauGaus")
    # hLandauGuasmaximum = hLandauGuas.GetBinContent( hLandauGuas.GetMaximumBin() )
    # hLandauGuas.Scale(hdEmaximum/hLandauGuasmaximum)
    # hLandauGuas.SetFillColorAlpha(ROOT.kGreen+2,0.1)
    # hLandauGuas.Draw("hist same")
    
    fitfunc = TF1("landau","[2]*TMath::Landau(x,[0],[1])",hdE_fixed.GetXaxis().GetXmin(),hdE_fixed.GetXaxis().GetXmax(),3)
    hdE_fixed.Fit(fitfunc,"MERS") 
    fitfunc.SetLineColor(ROOT.kBlue)
    fitfunc.Draw("same")
    
    cnv.SaveAs("dE_dixed.pdf")
    
    print(f"Model Gaus width: {Gwidth} [MeV]")
    print(f"Raw hist mean: {hdE_fixed.GetMean()} [MeV]")
    print(f"Raw hist MPV: {hdE_fixed.GetBinCenter( hdE_fixed.GetMaximumBin() )} [MeV]")
    print(f"GEANT4 mean dEdx: {par.getG4BBdEdx(100*U.MeV2eV)*U.eV2MeV} [MeV/cm]")
    print(f"GEANT4 mean x*dEdx: {(5*U.um2cm)*par.getG4BBdEdx(100*U.MeV2eV)*U.eV2MeV} [MeV]")
    print(f"Model mean: {par.Mean(100*U.MeV2eV, 5*U.um2cm)*U.eV2MeV} [MeV]")
    print(f"Model MPV: {par.MPV(100*U.MeV2eV, 5*U.um2cm)*U.eV2MeV} [MeV]")
    print(f"Model width: {par.Width(100*U.MeV2eV, 5*U.um2cm)*U.eV2MeV} [MeV]")
    print(f"Model hist mean: {hLandau.GetMean()} [MeV]")
    print(f"Model hist MPV: {hLandau.GetBinCenter( hLandau.GetMaximumBin() )} [MeV]")

    quit()
    

    E = 10*U.MeV2eV  # eV
    # E = 100*U.MeV2eV  # eV
    x = 5*U.um2cm     # cm
    # x = 300*U.um2cm     # cm
    n1 = par.n12_mean(E,x,1)
    n2 = par.n12_mean(E,x,2)
    n3 = par.n3_mean(E,x)
    
    n1t = par.n12_mpv(E,x,1)
    n2t = par.n12_mpv(E,x,2)
    n3t = par.n3_mpv(E,x)
    
    # Moment1 = par.Moment1(E)
    Delta_p = par.Delta_p(E,x)*U.eV2MeV
    Width   = par.Width(E,x)*U.eV2MeV
    print(f"Delta_p({E*U.eV2MeV} MeV, {x*U.cm2um} um)={Delta_p}, Width({E*U.eV2MeV} MeV, {x*U.cm2um} um)={Width}")
    # print(f"Moment1({E*U.eV2MeV} MeV, {x*U.cm2um} um)={Moment1}")
    print(f"n1({E*U.eV2MeV} MeV, {x*U.cm2um} um)={n1}, n2({E*U.eV2MeV} MeV, {x*U.cm2um} um)={n2}, n3({E*U.eV2MeV} MeV, {x*U.cm2um} um)={n3}")
    print(f"n1~({E*U.eV2MeV} MeV, {x*U.cm2um} um)={n1t}, n2~({E*U.eV2MeV} MeV, {x*U.cm2um} um)={n2t}, n3~({E*U.eV2MeV} MeV, {x*U.cm2um} um)={n3t}")
    
    higE_BB = par.BB(E)*U.eV2MeV
    lowE_BB = par.BBlowE(E)*U.eV2MeV
    print(f"higE_BB({E*U.eV2MeV})={higE_BB*U.eV2MeV}, lowE_BB({E*U.eV2MeV})={lowE_BB*U.eV2MeV}")
    
    h_dEdx_vs_E_higBB  = TH1D("h_dEdx_vs_E_higBB", ";E [MeV];dE/dx [MeV/cm]", len(bins.Ebins_forDp)-1,array.array("d",bins.Ebins_forDp))
    h_dEdx_vs_E_lowBB  = TH1D("h_dEdx_vs_E_lowBB", ";E [MeV];dE/dx [MeV/cm]", len(bins.Ebins_forDp)-1,array.array("d",bins.Ebins_forDp))
    for b in range(1,h_dEdx_vs_E_higBB.GetNbinsX()+1):
        E = h_dEdx_vs_E_higBB.GetBinCenter(b)*U.MeV2eV # eV
        higE_BB = par.BB(E)*U.eV2MeV
        lowE_BB = par.BBlowE(E)*U.eV2MeV
        h_dEdx_vs_E_higBB.SetBinContent( b, higE_BB )
        h_dEdx_vs_E_lowBB.SetBinContent( b, lowE_BB )
    h_dEdx_vs_E_higBB.SetLineColor(ROOT.kRed)
    h_dEdx_vs_E_lowBB.SetLineColor(ROOT.kBlack)
    leg = TLegend(0.5,0.74,0.8,0.88)
    leg.SetFillStyle(4000) # will be transparent
    leg.SetFillColor(0)
    leg.SetTextFont(42)
    leg.SetBorderSize(0)
    leg.AddEntry(h_dEdx_vs_E_higBB,"High-E BB","l")
    leg.AddEntry(h_dEdx_vs_E_lowBB,"Low-E BB","l")

    cnv = TCanvas("cnv","",500,1000)
    cnv.Divide(1,2)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    h_dEdx_vs_E_higBB.Draw("hist")
    h_dEdx_vs_E_lowBB.Draw("hist same")
    leg.Draw("same")
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    h_dEdx_vs_E_ratio = h_dEdx_vs_E_higBB.Clone()
    h_dEdx_vs_E_ratio.SetTitle("High-E/Low-E BB")
    h_dEdx_vs_E_ratio.Divide(h_dEdx_vs_E_lowBB)
    h_dEdx_vs_E_ratio.Draw("hist")
    cnv.SaveAs("dEdx.pdf")
    
    quit()
    
    # cnv = TCanvas("cnv","",500,500)
    # ROOT.gPad.SetTicks(1,1)
    # # ROOT.gPad.SetLogx()
    # ROOT.gPad.SetLogy()
    # f = fLandau(0,bins.dEbins[len(bins.dEbins)-1],[Delta_p, Width, 1])
    # # f = fLandauGaus(0,70,[1.25725, 20.8889, 11552.8, 4.0632])
    # f.Draw()
    # cnv.SaveAs("landau.pdf")

    # nEbins,Ebins = GetLogBinning(50,2,100)
    # ndxbins,dxbins = GetLogBinning(50,1e-7,1.5e2)
    x = 2*U.um2cm     # cm
    hDelta_p_vs_E  = TH1D("hDelta_p_vs_E", f"For a fixed x of {x*U.cm2um} [#mum]"+";E [MeV];#Delta_{p} [MeV]", len(bins.Ebins_forDp)-1,array.array("d",bins.Ebins_forDp))
    for b in range(1,hDelta_p_vs_E.GetNbinsX()+1):
        E = hDelta_p_vs_E.GetBinCenter(b)*U.MeV2eV # eV
        Delta_p = par.Delta_p(E,x)*U.eV2MeV
        hDelta_p_vs_E.SetBinContent( b, Delta_p )

    E = 100*U.MeV2eV  # eV
    hDelta_p_vs_dx = TH1D("hDelta_p_vs_dx",f"For a fixed E of {E*U.eV2MeV} [MeV]"+";dx [#mum];#Delta_{p} [MeV]", len(bins.dxbins_forDp)-1,array.array("d",bins.dxbins_forDp))
    for b in range(1,hDelta_p_vs_dx.GetNbinsX()+1):
        dx = hDelta_p_vs_dx.GetBinCenter(b)*U.um2cm
        Delta_p = par.Delta_p(E,dx)*U.eV2MeV
        hDelta_p_vs_dx.SetBinContent( b, Delta_p )

    x = 2*U.um2cm     # cm
    hDelta_p_vs_dx_vs_E = TH2D("hDelta_p_vs_dx_vs_E", ";E [MeV];dx [#mum];#Delta_{p} [MeV]", len(bins.Ebins_forDp)-1,array.array("d",bins.Ebins_forDp), len(bins.dxbins_forDp)-1,array.array("d",bins.dxbins_forDp))
    hWidth_vs_dx_vs_E   = TH2D("hWidth_vs_dx_vs_E",   ";E [MeV];dx [#mum];Width [MeV]", len(bins.Ebins_forDp)-1,array.array("d",bins.Ebins_forDp), len(bins.dxbins_forDp)-1,array.array("d",bins.dxbins_forDp))
    for bx in range(1,hDelta_p_vs_dx_vs_E.GetNbinsX()+1):
        E = hDelta_p_vs_E.GetXaxis().GetBinCenter(bx)*U.MeV2eV # eV
        for by in range(1,hDelta_p_vs_dx_vs_E.GetNbinsY()+1):
            dx = hDelta_p_vs_dx_vs_E.GetYaxis().GetBinCenter(by)*U.um2cm
            Delta_p = par.Delta_p(E,dx)*U.eV2MeV
            Width = par.Width(E,dx)*U.eV2MeV
            # print(f"E={E*U.eV2MeV}, dx={dx*U.cm2um}, Dp={Delta_p}")
            hDelta_p_vs_dx_vs_E.SetBinContent( bx,by, Delta_p )
            hWidth_vs_dx_vs_E.SetBinContent( bx,by, Width )

    cnv = TCanvas("cnv","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    hDelta_p_vs_E.Draw("hist")
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    hDelta_p_vs_dx.Draw("hist")
    cnv.SaveAs("Delta_p.pdf(")

    cnv = TCanvas("cnv","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    hDelta_p_vs_dx_vs_E.SetMinimum(1e-11)
    hDelta_p_vs_dx_vs_E.Draw("colz")
    cnv.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    hWidth_vs_dx_vs_E.Draw("colz")
    cnv.SaveAs("Delta_p.pdf)")

if __name__ == "__main__":
    main()