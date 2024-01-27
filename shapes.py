import array
import math
import numpy as np
import ROOT


class Functions:
    def __init__(self, name):
        self.name = name
        self.vars   = None
        self.de     = None
        self.ml     = None
        self.sl     = None
        self.mg     = None
        self.sg     = None
        self.landau = None
        self.gauss  = None

    def __str__(self):
        return f"{self.name}"

    def setpdfs(self,vars):
        self.de     = ROOT.RooRealVar("de", "#DeltaE [MeV]", vars["dE"][0], vars["dE"][1])
        self.ml     = ROOT.RooRealVar("ml", "mean landau",   vars["ml"][0], vars["ml"][1], vars["ml"][2])
        self.sl     = ROOT.RooRealVar("sl", "sigma landau",  vars["sl"][0], vars["sl"][1], vars["sl"][2])
        self.mg     = ROOT.RooRealVar("mg", "mg", 0)
        self.sg     = ROOT.RooRealVar("sg", "sg", vars["sg"][0], vars["sg"][1], vars["sg"][2])
        self.landau = ROOT.RooLandau("lx", "lx", self.de, self.ml, self.sl)
        self.gauss  = ROOT.RooGaussian("gauss", "gauss", self.de, self.mg, self.sg)
        # Set # of bins to be used for FFT sampling
        self.de.setBins(vars["nFFTbins"], "cache")
        # Construct landau (x) gauss
        self.convolution = ROOT.RooFFTConvPdf("lxg", "landau (X) gauss", self.de, self.landau, self.gauss)
        

    ### Landau(x)Gauss convolution https://root.cern/doc/master/langaus_8C.html
    def langaufun(self, x, par):
        print("1")
        
        # Fit parameters:
        # par[0]=Width (scale) parameter of Landau density
        # par[1]=Most Probable (MP, location) parameter of Landau density
        # par[2]=Total area (integral -inf to inf, normalization constant)
        # par[3]=Width (sigma) of convoluted Gaussian function
        # 
        # In the Landau distribution (represented by the CERNLIB approximation),
        # the maximum is located at x=-0.22278298 with the location parameter=0.
        # This shift is corrected within this function, so that the actual
        # maximum is identical to the MP parameter.
        
        # Numeric constants
        invsq2pi = 0.3989422804014 # (2 pi)^(-1/2)
        mpshift  = -0.22278298     # Landau maximum location
        
        # Control constants
        np = 100.0  # number of convolution steps
        sc =   5.0  # convolution extends to +-sc Gaussian sigmas
        
        # MP shift correction
        mpc = par[1] - mpshift * par[0]
        
        print("2")
        
        # Range of convolution integral
        xlow = x[0] - sc * par[3]
        xupp = x[0] + sc * par[3]
        
        step = (xupp-xlow) / np
        
        print("3")
        
        # Convolution integral of Landau and Gaussian by sum
        i = 1.0
        sum = 0.0
        while(i<=np/2):
            print(f"i1={i}")
            xx = xlow + (i-.5) * step
            fland = ROOT.TMath.Landau(xx,mpc,par[0]) / par[0]
            sum += fland * ROOT.TMath.Gaus(x[0],xx,par[3])
            print(f"i2={i}")
            xx = xupp - (i-.5) * step
            fland = ROOT.TMath.Landau(xx,mpc,par[0]) / par[0]
            sum += fland * ROOT.TMath.Gaus(x[0],xx,par[3])
            print(f"i3={i}")
            # propagate
            i += 1
        print("4")
        return (par[2] * step * sum * invsq2pi / par[3])

    ### get as TF1 given the range and the parameters 
    def fLandauGaus(self,xmin,xmax,par,name="LandauGaus"):
        npar = 4
        if(len(par)!=npar):
            print("in fLandauGaus: length of parameters list does not match, quitting.")
            quit()
        parname = ["Width", "MP", "Area", "GSigma"]
        parval  = par #[1.25725, 20.8889, 11552.8, 4.0632]
        f = ROOT.TF1(name,self.langaufun,xmin,xmax,npar)
        for i in range(npar):
            f.SetParameter(i,parval[i])
            f.SetParName(i,parname[i])
        return f
    
    ### get as TF1 given the range and the parameters 
    def fLandau(self,xmin,xmax,par,name="Landau"):
        npar = 3
        if(len(par)!=npar):
            print("in fLandau: length of parameters list does not match, quitting.")
            quit()
        parname = ["MP", "Width", "Area"]
        parval  = par
        f = ROOT.TF1(name,"[2]*TMath::Landau(x,[0],[1])",xmin,xmax,npar)
        for i in range(npar):
            f.SetParameter(i,parval[i])
            f.SetParName(i,parname[i])
        return f

    ### get as TF1 given the range and the parameters 
    def fGaus(self,xmin,xmax,par,name="Gauss"):
        npar = 3
        if(len(par)!=npar):
            print("in fGaus: length of parameters list does not match, quitting.")
            quit()
        parname = ["Mean", "Width", "Area"]
        parval  = par
        f = ROOT.TF1(name,"[2]*TMath::Gaus(x,[0],[1])",xmin,xmax,npar)
        for i in range(npar):
            f.SetParameter(i,parval[i])
            f.SetParName(i,parname[i])
        return f
    
    def f2h(self,f,h):
        name = f"{f.GetName()}_f2h"
        hnew = h.Clone(name)
        hnew.Reset()
        for b in range(1,hnew.GetNbinsX()+1):
            x = hnew.GetBinCenter(b)
            # w = hnew.GetXaxis().GetBinWidth(b)
            y = f.Eval(x)
            hnew.SetBinContent(b,y)
        hnew.SetLineColor( f.GetLineColor() )
        return hnew

    def asH(self,h,xmin,xmax,par,name):
        f = None
        if(name=="Landau"):     f = self.fLandau(xmin,xmax,par,"Landau")
        if(name=="Gaus"):       f = self.fGaus(xmin,xmax,par,"Gauss")
        if(name=="LandauGaus"): f = self.fLandauGaus(xmin,xmax,par,"LandauGaus")
        return self.f2h(f,h)
    
    
        