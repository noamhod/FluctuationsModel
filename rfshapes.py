import math
import array
import numpy as np
import ROOT
import units as U

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)

pi = 3.14159265359

class RooFitShapes:
    def __init__(self,name,model,hdE):
        self.name = name
        self.model = model
        self.isLandau = ("Landau" in self.model)
        self.isGauss  = ("Gauss" in self.model)
        self.isConv   = (self.isGauss and self.isLandau)
        self.hdE           = hdE
        self.data          = None
        self.dE            = None
        self.landau_mpv    = None
        self.landau_sigma  = None
        self.landau_model  = None
        self.gauss_mean    = None
        self.gauss_sigma   = None
        self.gauss_model   = None
        self.full_model    = None
        self.tf1           = {"Model":None,"Landau":None,"Gauss":None}
        self.model_title = "landau #otimes gauss convolution" if(self.isConv) else ""
        if(not self.isConv):
            self.model_title = "Gauss only" if(self.isGauss) else "Landau only"
        self.init_data()
        self.init_shapes()
        self.setTF1()


    def __str__(self):
        return f"{self.name}({self.model})"


    def borysov_ionization(self, y,par):
        ## par[0] = w3
        ## par[1] = w
        ## par[2] = p3 (lambda)
        ## par[3] = tMin
        ## par[4] = tMax
        a = str(par[0])
        b = str(par[0]/(1-par[1]))
        N = ROOT.TMath.Exp(-par[2])/(2*pi)
        A = "(cos("+a+"*x)/"+a+" - cos("+b+"*x)/"+b+" + x*ROOT::Math::sinint("+a+"*x) - x*ROOT::Math::sinint("+b+"*x))"
        B = "(-x*ROOT::Math::cosint("+a+"*x) + x*ROOT::Math::cosint("+b+"*x) + sin("+a+"*x)/"+a+" - sin("+b+"*x)/"+b+")"
        C = str(par[2]*par[0]/par[1])
        dE = str(y[0])
        F = "exp("+C+"*"+A+") * cos("+C+"*"+B+" - x*"+dE+")"
        f = ROOT.TF1("f",F,par[3],par[4])
        f.SetNpx(1000)
        upper_limit = par[4]
        if(y[0]>4500):  upper_limit /= 4
        I = f.Integral(par[3],upper_limit)
        return 0 if(y[0]>6000 or I<0) else N*I*2


    def borysov_excitation(self, x, par):
        ## par[0] = n1 (mean number of collisions of type 1)
        ## par[1] = e1 (type 1 excitation energy)
        nn = int(x[0]/par[1])
        # if(nn<0): return 0
        if(x[0]<0): nn -= 1
        xx = 0
        nnmax = nn+2
        for ii in range(nn,nnmax):
            if(ii<0): continue
            xx += math.pow(par[0],ii) / ROOT.TMath.Factorial(ii)
        xx *= 0.5*ROOT.TMath.Exp(-par[0])/par[1]
        return xx


    def truncated_gaus(self, x, par):
        ## par[0] = mean
        ## par[1] = sigma
        return ROOT.TMath.Gaus(x[0],par[0],par[1]) if(x[0]>0 and x[0]<2*par[0]) else 0

    def get_roopdf(self, name, pdfname, par, Nbins):
        f = None
        if(pdfname=="borysov_ionization"): f = ROOT.TF1("f"+name,borysov_ionization, self.hdE.GetXaxis().GetXmin(),self.hdE.GetXaxis().GetXmax(), len(par))
        if(pdfname=="borysov_excitation"): f = ROOT.TF1("f"+name,borysov_excitation, self.hdE.GetXaxis().GetXmin(),self.hdE.GetXaxis().GetXmax(), len(par))
        if(pdfname=="truncated_gaus"):     f = ROOT.TF1("f"+name,truncated_gaus,     self.hdE.GetXaxis().GetXmin(),self.hdE.GetXaxis().GetXmax(), len(par))
        for i in range(len(par)): f.SetParameter(i,par[i])
        f.SetLineColor(ROOT.kRed)
        f.SetNpx(1000)
        h = ROOT.TH1D("h"+name,"",Nbins, self.hdE.GetXaxis().GetXmin(),self.hdE.GetXaxis().GetXmax())
        for bb in range(1,h.GetNbinsX()+1):
            x = h.GetBinCenter(bb)
            y = f.Eval(x) #if(x>54) else 0 #TODO: is this cutoff physical?
            h.SetBinContent(bb,y)
        h.Scale(1./h.Integral())
        rdh = ROOT.RooDataHist("rdh"+name,"rdh"+name, self.dE, h)
        pdf = ROOT.RooHistPdf(name+"_model",name, self.dE, rdh)
        return pdf
    
    def get_rooconvpdf(self, name,title,pdf1,pdf2):
        conv = ROOT.RooFFTConvPdf(name+"_model", title, self.dE, pdf1, pdf2)
        return conv
        
    def get_full_model(self,name,title,pdf1,pdf2,pdf3):
        pdf12  = self.get_rooconvpdf(name+"_tmp",title+"_tmp",pdf1,pdf2)
        pdf123 = self.get_rooconvpdf(name+"_tmp",title+"_tmp",pdf12,pdf3)
        return pdf123
    
    def build_full_model(self, name, title, Nbins,
                               name1, pdfname1, par1,
                               name2, pdfname2, par2,
                               name3, pdfname3, par3):
        pdf1  = self.get_roopdf(name1, pdfname1, par1, Nbins)
        pdf2  = self.get_roopdf(name2, pdfname2, par2, Nbins)
        pdf3  = self.get_roopdf(name3, pdfname3, par3, Nbins)
        model = self.get_full_model(name,title,pdf1,pdf2,pdf3)
        tf1   = model.asTF(self.dE)
        return model,tf1
    
    def init_data(self):
        self.dE = ROOT.RooRealVar("dE", "#DeltaE [MeV]", self.hdE.GetXaxis().GetXmin(), self.hdE.GetXaxis().GetXmax())
        # rfbinning = ROOT.RooBinning(len(dEbins)-1,dEbins)
        self.dE.setBins(1000, "cache") ## set #bins to be used for FFT sampling to 10000
        ### RooFit data
        self.data = ROOT.RooDataHist("data","dataset",self.dE,self.hdE)

    def init_shapes(self):
        if(self.isLandau or self.isConv): self.landau_mpv   = ROOT.RooRealVar("landau_mpv", "mean landau", self.model["Landau"]["mpv"]*U.eV2MeV)     
        if(self.isLandau or self.isConv): self.landau_sigma = ROOT.RooRealVar("landau_sigma", "sigma landau", self.model["Landau"]["width"]*U.eV2MeV)
        if(self.isLandau or self.isConv): self.landau_model = ROOT.RooLandau("landau_model", "lx", self.dE, self.landau_mpv, self.landau_sigma)
        if(self.isGauss  or self.isConv): self.gauss_mean   = ROOT.RooRealVar("gauss_mean", "mg", self.model["Gauss"]["mean"]*U.eV2MeV)
        if(self.isGauss  or self.isConv): self.gauss_sigma  = ROOT.RooRealVar("gauss_sigma", "sg", self.model["Gauss"]["width"]*U.eV2MeV)
        if(self.isGauss  or self.isConv): self.gauss_model  = ROOT.RooGaussian("gauss_model", "gauss", self.dE, self.gauss_mean, self.gauss_sigma)
        if(self.isConv):                  self.full_model   = ROOT.RooFFTConvPdf("full_model", "landau(x)gauss", self.dE, self.landau_model, self.gauss_model)
        if(not self.isConv):
            self.full_model = self.landau_model if(self.isLandau) else self.gauss_model

    def plot(self):
        frame = self.dE.frame(ROOT.RooFit.Title(self.model_title))
        # data.plotOn(frame, ROOT.RooFit.Binning(rfbinning), ROOT.RooFit.Name("data"))
        self.data.plotOn(frame, ROOT.RooFit.Binning(self.hdE.GetNbinsX()), ROOT.RooFit.Name("data"))
        if(self.isLandau): self.landau_model.plotOn(frame, ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Name("landau_model"))
        if(self.isGauss):  self.gauss_model.plotOn(frame, ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("gauss_model"))
        if(self.isConv):   self.full_model.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Name("full_model"))
        return frame
    
    def getTF1(self):
        return self.tf1
    
    def setTF1(self):
        if(self.isConv):
            self.tf1["Model"]  = self.full_model.asTF(self.dE, ROOT.RooArgList(self.landau_mpv,self.landau_sigma,self.gauss_mean,self.gauss_sigma), self.dE)
            self.tf1["Landau"] = self.full_model.asTF(self.dE, ROOT.RooArgList(self.landau_mpv,self.landau_sigma), self.dE)
            self.tf1["Gauss"]  = self.full_model.asTF(self.dE, ROOT.RooArgList(self.gauss_mean,self.gauss_sigma), self.dE)
        else:
            if(self.isLandau):
                self.tf1["Model"] = self.full_model.asTF(self.dE, ROOT.RooArgList(self.landau_mpv,self.landau_sigma), self.dE)
            if(self.isGauss):
                self.tf1["Model"] = self.full_model.asTF(self.dE, ROOT.RooArgList(self.gauss_mean,self.gauss_sigma), self.dE)
    
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