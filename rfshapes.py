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
        self.model_title = "landau #otimes gauss convolution" if(self.isConv) else ""
        if(not self.isConv):
            self.model_title = "Gauss only" if(self.isGauss) else "Landau only"
        self.init_data()
        self.init_shapes()

    def __str__(self):
        return f"{self.name}({self.model})"
    
    def init_data(self):
        self.dE = ROOT.RooRealVar("dE", "#DeltaE [MeV]", self.hdE.GetXaxis().GetXmin(), self.hdE.GetXaxis().GetXmax())
        # rfbinning = ROOT.RooBinning(len(dEbins)-1,dEbins)
        self.dE.setBins(100000, "cache") ## set #bins to be used for FFT sampling to 10000
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
